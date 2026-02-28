#!/bin/bash
# Qwen3-TTS model conversion pipeline
# Run on x86 machine with 24GB+ RAM
#
# Usage:
#   ./convert_all.sh --hf-model Qwen/Qwen3-TTS --output ./models --target rk3588
#
# Prerequisites:
#   uv installed, RKNN toolkit2 packages available

set -euo pipefail

HF_MODEL="Qwen/Qwen3-TTS"
OUTPUT="./models"
TARGET="rk3588"
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-model) HF_MODEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Qwen3-TTS Model Conversion"
echo "  HF Model:  $HF_MODEL"
echo "  Output:    $OUTPUT"
echo "  Target:    $TARGET"
echo "============================================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTPUT"/{embeddings,code_predictor,vocoder}

# Create a virtual environment for conversion
echo "[1/6] Setting up Python environment..."
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$SCRIPT_DIR"
uv venv --python 3.10 .venv 2>/dev/null || true
source .venv/bin/activate

# Install conversion dependencies
uv pip install torch transformers safetensors numpy onnx onnxsim 2>/dev/null

# Install RKNN toolkit2 for x86
RKNN_PKG="rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
if [ -f "$RKNN_PKG" ]; then
    uv pip install "$RKNN_PKG" 2>/dev/null
else
    echo "Note: RKNN toolkit2 wheel not found locally."
    echo "Download from: https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages"
    echo "Vocoder RKNN conversion will be skipped."
fi

# Step 1: Extract talker as Qwen3
echo
echo "[2/6] Extracting talker weights..."
python3 "$SCRIPT_DIR/extract_talker.py" \
    --hf_model "$HF_MODEL" \
    --output "$OUTPUT/talker_qwen3"

# Step 2: Extract embeddings
echo
echo "[3/7] Extracting embeddings..."
python3 "$SCRIPT_DIR/extract_embeddings.py" \
    --hf_model "$HF_MODEL" \
    --output "$OUTPUT/embeddings"

# Step 2b: Download tokenizer.json
echo
echo "[4/7] Downloading tokenizer..."
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$HF_MODEL', trust_remote_code=True)
t.save_pretrained('$OUTPUT')
print('tokenizer.json saved')
"

# Step 3: Export code predictor
echo
echo "[4/6] Exporting code predictor..."
python3 "$SCRIPT_DIR/export_code_predictor.py" \
    --hf_model "$HF_MODEL" \
    --output "$OUTPUT/code_predictor"

# Step 4: Export vocoder to ONNX
echo
echo "[5/6] Exporting vocoder to ONNX..."
python3 "$SCRIPT_DIR/export_vocoder.py" \
    --hf_model "$HF_MODEL" \
    --output "$OUTPUT/vocoder" \
    --codes_length 64

# Step 5: Convert to target format
echo
echo "[6/6] Converting to $TARGET format..."

# Convert talker to GGUF (for llama.cpp)
if command -v llama-gguf-convert &> /dev/null || [ -f "$SCRIPT_DIR/convert_talker_gguf.py" ]; then
    echo "  Converting talker to GGUF..."
    python3 "$SCRIPT_DIR/convert_talker_gguf.py" \
        --input "$OUTPUT/talker_qwen3" \
        --output "$OUTPUT/qwen3_tts_talker_q4_k_m.gguf" 2>/dev/null || \
        echo "  GGUF conversion requires llama.cpp convert tools"
fi

# Convert vocoder to RKNN (if toolkit available)
if python3 -c "from rknn.api import RKNN" 2>/dev/null; then
    echo "  Converting vocoder to RKNN..."
    python3 "$SCRIPT_DIR/convert_vocoder_rknn.py" \
        --onnx "$OUTPUT/vocoder/vocoder_traced_64.onnx" \
        --output "$OUTPUT/vocoder/vocoder_traced_64_q8.rknn" \
        --target "$TARGET"
else
    echo "  Skipping RKNN conversion (toolkit not installed)"
fi

echo
echo "============================================================"
echo "  Conversion complete!"
echo "  Models saved to: $OUTPUT"
echo ""
echo "  Deploy to RK3588 machines:"
echo "    scp -r $OUTPUT/ user@machine1:~/models/"
echo "    scp -r $OUTPUT/ user@machine2:~/models/"
echo "============================================================"
