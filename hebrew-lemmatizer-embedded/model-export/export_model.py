#!/usr/bin/env python3
"""
Export DictaBERT-Lex to ONNX format for the Hebrew Lemmatizer plugin.

This script:
1. Downloads the dicta-il/dictabert-lex model from HuggingFace
2. Converts it to ONNX format with INT8 weight quantization
3. Replaces the full logits output with the sorted top-three token IDs
4. Prunes the output head to the vocabulary entries that are ever returned as a lemma
5. Copies the tokenizer.json for use by the Java plugin
"""

import os
import shutil
import json
import hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from optimum.onnxruntime import ORTModelForMaskedLM
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODEL_NAME = "dicta-il/dictabert-lex"
OUTPUT_DIR = Path(__file__).parent.parent / "plugin-lemmas-embedded" / "src" / "main" / "resources" / "model"
HEAD_KEEP_PATH = Path(__file__).parent / "head-keep-48k.txt"


def calculate_cache_key(*paths: Path) -> str:
    """Build a content key so cached model resources cannot survive an incompatible upgrade."""
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def add_topk_output(input_path: Path, output_path: Path, k: int = 3):
    """Keep model predictions identical while avoiding a huge logits transfer to Java."""
    model = onnx.load(str(input_path))
    if len(model.graph.output) != 1:
        raise ValueError("Expected the exported model to have exactly one logits output")

    logits_name = model.graph.output[0].name
    k_name = "lemma_topk_k"
    values_name = "lemma_topk_values"
    indices_name = "lemma_topk_indices"

    model.graph.initializer.append(
        helper.make_tensor(k_name, TensorProto.INT64, [1], [k])
    )
    model.graph.node.append(
        helper.make_node(
            "TopK",
            inputs=[logits_name, k_name],
            outputs=[values_name, indices_name],
            name="LemmaTopK",
            axis=-1,
            largest=1,
            sorted=1,
        )
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(
            indices_name,
            TensorProto.INT64,
            ["batch_size", "sequence_length", k],
        )
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def load_head_keep_ids(path: Path = HEAD_KEEP_PATH) -> np.ndarray:
    ids = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return np.array(sorted(set(ids)), dtype=np.int64)


def prune_output_head(model_path: Path, keep_ids: np.ndarray):
    """Keep only the given vocabulary columns of the output head.

    The head scores every vocabulary entry at every position, but only a fraction of them is ever
    returned as a lemma. Slicing the weight, its per-channel zero point and scale, and the bias to the
    kept columns makes the head several times smaller and faster. The top-three indices are mapped back
    to the original vocabulary ids, so the plugin needs no change and still sees the same output.
    """
    model = onnx.load(str(model_path))
    graph = model.graph
    nodes = list(graph.node)
    initializers = {init.name: init for init in graph.initializer}

    head = next(n for n in nodes if n.op_type == "MatMulInteger" and n.name.endswith("decoder/MatMul_quant"))
    weight = numpy_helper.to_array(initializers[head.input[1]])
    vocab_size = weight.shape[1]
    if keep_ids.min() < 0 or keep_ids.max() >= vocab_size:
        raise ValueError(f"Head keep list does not fit the {vocab_size}-entry vocabulary")

    def replace(name, array):
        for i, init in enumerate(graph.initializer):
            if init.name == name:
                graph.initializer[i].CopyFrom(numpy_helper.from_array(array, name))
                return
        raise KeyError(name)

    replace(head.input[1], np.ascontiguousarray(weight[:, keep_ids]))
    zero_point = numpy_helper.to_array(initializers[head.input[3]])
    if zero_point.ndim == 1:
        replace(head.input[3], zero_point[keep_ids])

    scales = next(n for n in nodes if n.name.endswith("decoder/MatMul_quant_scales_mul"))
    bias = next(n for n in nodes if n.name.endswith("decoder/Add"))
    for node in (scales, bias):
        for name in node.input:
            if name in initializers and list(initializers[name].dims) == [vocab_size]:
                replace(name, numpy_helper.to_array(initializers[name])[keep_ids])

    topk = next(n for n in nodes if n.op_type == "TopK")
    final_indices = topk.output[1]
    topk.output[1] = "pruned_topk_indices"
    graph.initializer.append(numpy_helper.from_array(keep_ids, "head_keep_ids"))
    nodes.append(helper.make_node("Gather", ["head_keep_ids", "pruned_topk_indices"], [final_indices], name="HeadKeepRemap"))
    del graph.node[:]
    graph.node.extend(nodes)

    # Shapes recorded for the resized head are stale now.
    value_info = [vi for vi in graph.value_info
                  if "decoder" not in vi.name and "topk" not in vi.name.lower() and vi.name != final_indices]
    del graph.value_info[:]
    graph.value_info.extend(value_info)

    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    print(f"Pruned output head from {vocab_size} to {len(keep_ids)} vocabulary entries")


def export_model():
    """Export the model to ONNX format."""
    print(f"Exporting {MODEL_NAME} to ONNX...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Save tokenizer.json
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    print(f"Saving tokenizer to {tokenizer_path}")
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Write vocab.txt for pure Java WordPiece tokenizer
    vocab_path = OUTPUT_DIR / "vocab.txt"
    vocab = tokenizer.get_vocab()
    print(f"Writing vocab to {vocab_path} ({len(vocab)} entries)")
    with vocab_path.open("w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda item: item[1]):
            f.write(token + "\n")
    
    # The HuggingFace tokenizer saves multiple files, we mainly need tokenizer.json
    # Rename if saved with different name
    if not tokenizer_path.exists():
        # Try to find the tokenizer file
        for f in OUTPUT_DIR.glob("*.json"):
            if "tokenizer" in f.name.lower():
                print(f"Found tokenizer: {f}")
                break
    
    # Export model to ONNX using optimum
    print("Converting model to ONNX...")
    
    # Create a temporary directory for the ONNX export
    temp_dir = Path(__file__).parent / "temp_onnx"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Export using optimum
        ort_model = ORTModelForMaskedLM.from_pretrained(
            MODEL_NAME,
            export=True,
            provider="CPUExecutionProvider",
            dtype=torch.float32
        )
        
        # Save the ONNX model
        ort_model.save_pretrained(str(temp_dir))
        
        # Locate the exported ONNX model file
        onnx_file = temp_dir / "model.onnx"
        if not onnx_file.exists():
            # Look for any .onnx file
            for f in temp_dir.glob("*.onnx"):
                onnx_file = f
                break

        if not onnx_file.exists():
            raise FileNotFoundError("ONNX export did not produce a model file")

        # Quantize to INT8 for fast CPU inference
        quantized_file = temp_dir / "model-int8.onnx"
        print(f"Quantizing ONNX model to INT8: {quantized_file}")
        quantize_dynamic(str(onnx_file), str(quantized_file), weight_type=QuantType.QInt8)

        # Return only the top-three IDs. The Java filter never uses the remaining
        # logits, and exporting them creates a large native-to-heap copy per field.
        output_model = OUTPUT_DIR / "model.onnx"
        print("Adding sorted top-three output to the ONNX graph")
        add_topk_output(quantized_file, output_model)
        print("Pruning the output head")
        prune_output_head(output_model, load_head_keep_ids())
        print(f"Saved quantized ONNX model to {output_model}")

        cache_key = calculate_cache_key(output_model, tokenizer_path)
        cache_key_path = OUTPUT_DIR / "model-cache-key.txt"
        cache_key_path.write_text(cache_key + "\n", encoding="utf-8")
        print(f"Saved model cache key to {cache_key_path}")
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    # Print model info
    model_path = OUTPUT_DIR / "model.onnx"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nModel exported successfully!")
        print(f"  Model size: {size_mb:.1f} MB")
        print(f"  Output directory: {OUTPUT_DIR}")
    else:
        print("ERROR: Model export failed!")
        return False
    
    # List output files
    print("\nOutput files:")
    for f in OUTPUT_DIR.iterdir():
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")
    
    return True


def test_model():
    """Test the exported ONNX model."""
    import onnxruntime as ort
    
    print("\nTesting ONNX model...")
    
    model_path = OUTPUT_DIR / "model.onnx"
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    
    # Create ONNX session
    session = ort.InferenceSession(str(model_path))
    
    # Print model inputs/outputs
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    
    # Test with Hebrew text
    test_text = "הלכתי"
    print(f"\nTest input: {test_text}")
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="np")
    
    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", inputs["attention_mask"] * 0)
        }
    )
    
    print(f"Output shape: {outputs[0].shape}")
    
    top_predictions = outputs[0][0]  # First batch

    # Get predictions for position 1 (first token after [CLS])
    if len(top_predictions) > 1:
        print("Top 3 predictions:")
        for idx in top_predictions[1]:
            token = tokenizer.decode([idx])
            print(f"  {idx}: {token}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    if export_model():
        test_model()
