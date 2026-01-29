#!/bin/bash
# Build script for Hebrew Lemmatizer Embedded Plugin
# Requires: Java 21, Python 3.8+

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Hebrew Lemmatizer Embedded Plugin Build ==="
echo ""

# Check Java version
if ! command -v java &> /dev/null; then
    echo "ERROR: Java not found. Please install Java 21."
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
echo "Java version: $JAVA_VERSION"

if [ "$JAVA_VERSION" -lt 21 ]; then
    echo "WARNING: Java 21+ recommended for ES 9.x"
fi

# Check for model files
MODEL_DIR="plugin-lemmas-embedded/src/main/resources/model"
if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    echo ""
    echo "=== Step 1: Export ONNX Model ==="
    echo ""

    cd model-export

    # Create virtual environment if needed
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    # Activate and install dependencies
    source venv/bin/activate
    pip install -q -r requirements.txt

    # Export model
    echo "Exporting DictaBERT to ONNX format..."
    python export_model.py

    deactivate
    cd "$SCRIPT_DIR"
fi

# Check model exists now
if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    echo "ERROR: Model file not found at $MODEL_DIR/model.onnx"
    echo "Please run the export script manually:"
    echo "  cd model-export && python export_model.py"
    exit 1
fi

echo ""
echo "=== Step 2: Build Plugin ==="
echo ""

cd plugin-lemmas-embedded

# Download Gradle wrapper if not present
if [ ! -f "gradlew" ]; then
    echo "Setting up Gradle wrapper..."

    # Download and extract Gradle
    GRADLE_VERSION="8.5"
    if [ ! -d "/tmp/gradle-${GRADLE_VERSION}" ]; then
        echo "Downloading Gradle ${GRADLE_VERSION}..."
        curl -sL "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -o /tmp/gradle.zip
        unzip -q /tmp/gradle.zip -d /tmp/
        rm /tmp/gradle.zip
    fi

    export PATH="/tmp/gradle-${GRADLE_VERSION}/bin:$PATH"
    gradle wrapper
fi

# Build the plugin
echo "Building plugin..."
./gradlew clean bundlePlugin

# Find and report the output
PLUGIN_ZIP=$(find build/distributions -name "*.zip" 2>/dev/null | head -n 1)

if [ -n "$PLUGIN_ZIP" ]; then
    echo ""
    echo "=== Build Complete ==="
    echo ""
    echo "Plugin: $PLUGIN_ZIP"
    echo "Size: $(du -h "$PLUGIN_ZIP" | cut -f1)"
    echo ""
    echo "Installation:"
    echo "  bin/elasticsearch-plugin install file://$(pwd)/$PLUGIN_ZIP"
else
    echo "ERROR: Plugin ZIP not found"
    exit 1
fi
