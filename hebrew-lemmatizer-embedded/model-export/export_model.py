#!/usr/bin/env python3
"""
Export DictaBERT-tiny-joint model to ONNX format for the Hebrew Lemmatizer plugin.

This script:
1. Downloads the dicta-il/dictabert-tiny-joint model from HuggingFace
2. Converts it to ONNX format with FP16 optimization
3. Copies the tokenizer.json for use by the Java plugin
"""

import os
import shutil
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from optimum.onnxruntime import ORTModelForMaskedLM
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_NAME = "dicta-il/dictabert-tiny-joint"
OUTPUT_DIR = Path(__file__).parent.parent / "plugin-lemmas-embedded" / "src" / "main" / "resources" / "model"


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
            provider="CPUExecutionProvider"
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

        # Copy quantized model into plugin resources
        shutil.copy(quantized_file, OUTPUT_DIR / "model.onnx")
        print(f"Saved quantized ONNX model to {OUTPUT_DIR / 'model.onnx'}")
        
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
    
    # Get top prediction for the word
    import numpy as np
    logits = outputs[0][0]  # First batch
    
    # Get predictions for position 1 (first token after [CLS])
    if len(logits) > 1:
        token_logits = logits[1]
        top_ids = np.argsort(token_logits)[-5:][::-1]
        
        print("Top 5 predictions:")
        for idx in top_ids:
            token = tokenizer.decode([idx])
            print(f"  {idx}: {token} ({token_logits[idx]:.2f})")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    if export_model():
        test_model()
