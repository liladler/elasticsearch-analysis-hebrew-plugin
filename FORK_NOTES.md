# Fork Notes: Embedded ONNX Hebrew Lemmatizer

This fork embeds the Dicta model inside the Elasticsearch plugin (no Docker service), uses ONNX Runtime for in-process inference, and includes an INT8-quantized ONNX model for fast CPU.

## What was done

- Exported DictaBERT (`dicta-il/dictabert-tiny-joint`) to ONNX and added INT8 dynamic quantization.
- Embedded the model artifacts inside the plugin resources.
- Converted the lemmatizer plugin to a **classic** ES 9.x plugin with JPMS module naming.
- Added entitlements for native library loading and data-path write access.
- Patched ONNX Runtime to remove `Runtime.addShutdownHook` (required for ES 9.x entitlements).

## Key paths

- Model export script: `hebrew-lemmatizer-embedded/model-export/export_model.py`
- Plugin build: `hebrew-lemmatizer-embedded/plugin-lemmas-embedded/build.gradle`
- Entitlements: `hebrew-lemmatizer-embedded/plugin-lemmas-embedded/src/main/plugin-metadata/entitlement-policy.yaml`
- Model resources: `hebrew-lemmatizer-embedded/plugin-lemmas-embedded/src/main/resources/model/`

## Build a new plugin for a newer ES version

1) Update versions in `hebrew-lemmatizer-embedded/plugin-lemmas-embedded/build.gradle`:

- `ext.elasticsearchVersion`
- `ext.luceneVersion` (match the ES version)
- (Optional) `ext.onnxRuntimeVersion` if you want a newer runtime

2) Re-export and INT8-quantize the model:

```
cd hebrew-lemmatizer-embedded/model-export
python3 export_model.py
```

3) Build a **Linux** plugin zip (recommended for Elastic Cloud):

```
docker run --rm \
  -v "/Users/lilyadler/git/hebrew-lemmatizer-4ElasticSearch_v1:/workspace" \
  -w /workspace/hebrew-lemmatizer-embedded/plugin-lemmas-embedded \
  eclipse-temurin:21-jdk ./gradlew clean bundlePlugin
```

4) Install the zip into ES and restart:

```
/path/to/elasticsearch/bin/elasticsearch-plugin remove heb-lemmas-embedded-plugin
/path/to/elasticsearch/bin/elasticsearch-plugin install file:///path/to/heb-lemmas-embedded-plugin-2.0-SNAPSHOT.zip
/path/to/elasticsearch/bin/elasticsearch
```

5) Verify with `_analyze`:

```
curl -k -X POST "https://localhost:9200/_analyze" \
  -H "Content-Type: application/json" \
  -u "elastic:<password>" \
  -d '{"tokenizer":"whitespace","filter":["heb_lemmas","heb_stopwords"],"text":"האזרחים שבבתיהם"}'
```

## Notes

- The model is **INT8 quantized** (dynamic quantization) via `onnxruntime.quantization.quantize_dynamic`.
- Entitlements are bundled in the plugin zip under `plugin-metadata/entitlement-policy.yaml`.
- The ONNX Runtime jar is patched at build time to avoid entitlement violations.
