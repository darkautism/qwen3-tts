#!/bin/bash
# Deploy models and worker to RK3588 machines
#
# Usage:
#   ./deploy.sh --models ./models --talker-host localhost --predictor-host 192.168.50.171

set -euo pipefail

MODELS_DIR="./models"
TALKER_HOST=""
PREDICTOR_HOST=""
REMOTE_USER="kautism"
REMOTE_DIR="~/qwen3-tts"

while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODELS_DIR="$2"; shift 2 ;;
        --talker-host) TALKER_HOST="$2"; shift 2 ;;
        --predictor-host) PREDICTOR_HOST="$2"; shift 2 ;;
        --user) REMOTE_USER="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

deploy_to() {
    local host="$1"
    local role="$2"
    echo "Deploying to $host ($role)..."

    # Create remote directory
    ssh "$REMOTE_USER@$host" "mkdir -p $REMOTE_DIR/{models,worker}"

    # Copy worker code
    scp "$PROJECT_DIR/worker/worker_server.py" "$REMOTE_USER@$host:$REMOTE_DIR/worker/"
    scp "$PROJECT_DIR/worker/pyproject.toml" "$REMOTE_USER@$host:$REMOTE_DIR/worker/"

    # Copy models based on role
    if [ "$role" = "talker" ]; then
        echo "  Copying talker models..."
        ssh "$REMOTE_USER@$host" "mkdir -p $REMOTE_DIR/models/embeddings"
        scp "$MODELS_DIR"/embeddings/*.npy "$REMOTE_USER@$host:$REMOTE_DIR/models/embeddings/"
        # Copy talker model (GGUF or RKLLM)
        for f in "$MODELS_DIR"/*.gguf "$MODELS_DIR"/*.rkllm; do
            [ -f "$f" ] && scp "$f" "$REMOTE_USER@$host:$REMOTE_DIR/models/"
        done
    elif [ "$role" = "predictor" ]; then
        echo "  Copying predictor + vocoder models..."
        ssh "$REMOTE_USER@$host" "mkdir -p $REMOTE_DIR/models/{code_predictor,vocoder,embeddings}"
        scp "$MODELS_DIR"/code_predictor/* "$REMOTE_USER@$host:$REMOTE_DIR/models/code_predictor/"
        scp "$MODELS_DIR"/vocoder/* "$REMOTE_USER@$host:$REMOTE_DIR/models/vocoder/"
        # Also need embeddings for feedback computation
        scp "$MODELS_DIR"/embeddings/codec_embedding.npy "$REMOTE_USER@$host:$REMOTE_DIR/models/embeddings/"
        scp "$MODELS_DIR"/embeddings/text_embedding.npy "$REMOTE_USER@$host:$REMOTE_DIR/models/embeddings/"
        scp "$MODELS_DIR"/embeddings/text_projection_*.npy "$REMOTE_USER@$host:$REMOTE_DIR/models/embeddings/"
    fi

    # Install Python dependencies
    echo "  Installing dependencies..."
    ssh "$REMOTE_USER@$host" "cd $REMOTE_DIR/worker && uv pip install msgpack numpy onnxruntime transformers 2>/dev/null || pip install msgpack numpy onnxruntime transformers"

    echo "  Done: $host ($role)"
}

if [ -n "$TALKER_HOST" ]; then
    deploy_to "$TALKER_HOST" "talker"
fi

if [ -n "$PREDICTOR_HOST" ]; then
    deploy_to "$PREDICTOR_HOST" "predictor"
fi

echo
echo "============================================================"
echo "  Deployment complete!"
echo
echo "  Start workers:"
if [ -n "$TALKER_HOST" ]; then
    echo "    ssh $REMOTE_USER@$TALKER_HOST 'cd $REMOTE_DIR && python worker/worker_server.py --role talker --models ./models --bind 0.0.0.0:9090'"
fi
if [ -n "$PREDICTOR_HOST" ]; then
    echo "    ssh $REMOTE_USER@$PREDICTOR_HOST 'cd $REMOTE_DIR && python worker/worker_server.py --role predictor --models ./models --bind 0.0.0.0:9090'"
fi
echo "============================================================"
