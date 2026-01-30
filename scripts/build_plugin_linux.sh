#!/usr/bin/env bash
set -euo pipefail

ES_VERSION="9.2.4"
ONNX_VERSION=""

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="${ROOT_DIR}/hebrew-lemmatizer-embedded/plugin-lemmas-embedded"
MODEL_EXPORT_DIR="${ROOT_DIR}/hebrew-lemmatizer-embedded/model-export"
ZIP_PATH="${PLUGIN_DIR}/build/distributions/heb-lemmas-embedded-plugin-2.0-SNAPSHOT.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --es-version)
      ES_VERSION="${2:-}"
      shift 2
      ;;
    --onnx-version)
      ONNX_VERSION="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--es-version <x.y.z>] [--onnx-version <x.y.z>]"
      exit 1
      ;;
  esac
done

ES_ZIP_PATH="${PLUGIN_DIR}/build/distributions/heb-lemmas-embedded-plugin-${ES_VERSION}.zip"

echo "==> Exporting INT8 ONNX model"
cd "${MODEL_EXPORT_DIR}"
python3 export_model.py

echo "==> Building plugin zip (Linux)"
GRADLE_ARGS=()
if [[ -n "${ES_VERSION}" ]]; then
  GRADLE_ARGS+=("-PelasticsearchVersion=${ES_VERSION}")
fi
if [[ -n "${ONNX_VERSION}" ]]; then
  GRADLE_ARGS+=("-PonnxRuntimeVersion=${ONNX_VERSION}")
fi

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace/hebrew-lemmatizer-embedded/plugin-lemmas-embedded \
  eclipse-temurin:21-jdk ./gradlew clean bundlePlugin "${GRADLE_ARGS[@]}"

echo "==> Renaming zip to include ES version"
cp -f "${ZIP_PATH}" "${ES_ZIP_PATH}"

echo "==> Done"
echo "Zip: ${ES_ZIP_PATH}"
