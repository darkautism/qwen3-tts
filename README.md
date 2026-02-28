<div align="center">

# qwen3-tts

Distributed Qwen3-TTS speech synthesis system — a Rust implementation designed for RK3588 clusters.

Single binary serves as CLI, OpenAI-compatible API server, MCP server, and inference worker.
Models auto-download from HuggingFace Hub.
**Zero Python dependencies** — all inference uses Candle (Rust ML framework), voice encoding is native ARM64.
Single statically-compiled binary; Talker/Predictor require no external `.so` libraries.

[![][license-shield]][license-shield-link]
[![][last-commit-shield]][last-commit-shield-link]

[中文文檔](README-zh.md)

</div>

## Architecture

```
 Machine 1 (IP1 - RK3588)              Machine 2 (IP2 - RK3588)
┌──────────────────────────┐          ┌──────────────────────────┐
│  qwen3-tts (Rust)        │          │  Worker: Predictor       │
│  ├── CLI / API / MCP     │   TCP    │  ├── CodePredictor       │
│  └── Orchestrator        │◄───────►│  │   (Candle Q8, CPU)    │
├──────────────────────────┤          │  │              :9091    │
│  Worker: Talker    :9090 │          ├──────────────────────────┤
│  ├── Tokenizer (HF)      │          │  Worker: Vocoder         │
│  ├── TextEmbedder (npy)  │          │  └── Vocoder             │
│  └── Talker LLM          │          │      (ONNX, CPU)  :9092 │
│      (Candle GGUF, CPU)  │          └──────────────────────────┘
└──────────────────────────┘
```

**Three Worker Roles:**

| Worker | Function | Compute | Default Port |
|--------|----------|---------|--------------|
| **Talker** | Tokenizer + TextEmbedder + LLM | CPU A76 × 4 | 9090 |
| **Predictor** | CodePredictor (Candle GGUF Q8_0) + feedback embedding | CPU A76 × 4 | 9091 |
| **Vocoder** | Vocoder (ONNX FP32 CPU, or RKNN INT8 NPU) | CPU/NPU | 9092 |

Each RK3588 saturates its CPU cores. Token generation: Candle ~3.6 tok/s (default) / GGML ~4.0 tok/s (`--features ggml-backend`).

## Requirements

### Hardware
- 1–3 RK3588 boards (16GB+ RAM recommended)

### Runtime Dependencies

Inference core (Talker, Predictor) uses Candle (pure Rust) — **no external libraries needed**.

**Vocoder machine requires:**

| Library | Source | Install Path |
|---------|--------|-------------|
| `libonnxruntime.so` | `pip install onnxruntime` (auto-detected) | Python package or system path |

> With `--features rknn-vocoder`, requires `librknnrt.so` (RKNN Runtime) instead.

### Installing ONNX Runtime (Vocoder Machine)

```bash
# Easiest: via Python package (auto-detected by the binary)
pip install onnxruntime

# Or manual system install
# Download aarch64 build: https://github.com/microsoft/onnxruntime/releases
sudo cp libonnxruntime.so /usr/lib/
```

### (Optional) RKNN Runtime

Only needed with `--features rknn-vocoder`:

```bash
sudo curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so \
  -o /lib/librknnrt.so
```

## Building

```bash
# Standard build (Candle inference + ONNX FP32 vocoder — pure Rust, no extra .so)
cargo build --release

# C++ GGML backend (ARM NEON SDOT acceleration, ~2x faster than Candle, needs llama_wrapper.so)
cargo build --release --features ggml-backend

# RKNN INT8 vocoder (NPU acceleration, has quantization noise, needs librknnrt.so)
cargo build --release --features rknn-vocoder

# Both enabled
cargo build --release --features ggml-backend,rknn-vocoder
# Output: target/release/qwen3-tts (~15-20 MB)
```

> Cross-compile: `cross build --release --target aarch64-unknown-linux-gnu`

### Feature Gates

| Feature | Description | Extra Dependencies | Performance |
|---------|-------------|-------------------|-------------|
| (default) | Candle inference + ONNX vocoder | `libonnxruntime.so` | ~3.6 tok/s |
| `ggml-backend` | C++ GGML/llama.cpp inference | `llama_wrapper.so` + `libllama.so` + `libggml*.so` | **~4.0 tok/s** |
| `rknn-vocoder` | RKNN INT8 vocoder (NPU) | `librknnrt.so` + RKNPU kernel | — |

Default uses Candle (pure Rust) inference — no C/C++ library installation needed.
Enabling `ggml-backend` leverages ARM NEON SDOT hardware instructions for ~10-15% extra speed.
The Candle backend already includes SDOT inline assembly optimization + pre-allocated memory pools.

## Quick Start

### Initialize Configuration

```bash
# Generate qwen3-tts.toml with your worker IPs
qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP2>
```

### Start Workers (models auto-download from HF Hub)

```bash
# IP1 - Talker Worker
qwen3-tts worker -r talker -b 0.0.0.0:9090

# IP2 - Predictor Worker
qwen3-tts worker -r predictor -b 0.0.0.0:9091

# IP2 - Vocoder Worker (can share machine with Predictor)
qwen3-tts worker -r vocoder -b 0.0.0.0:9092
```

> Models download to `~/.local/share/qwen3-tts/models/{role}/` by default.
> Custom HF repo: `--repo your-name/your-repo`

### Synthesize Speech

```bash
# Simple usage (outputs output.wav)
qwen3-tts "你好世界"

# Specify output and language
qwen3-tts speak "Hello world" -o speech.wav --lang english

# Voice cloning (custom voice file)
qwen3-tts speak "你好" --voice my_voice.json -o clone.wav
```

## Voice Cloning

### Creating a Voice Profile

Runs directly on RK3588 — **no x86 or Python needed**:

```bash
# Encode reference audio → outputs a single .json voice file (any sample rate, auto-resampled to 24kHz)
qwen3-tts encode-voice \
    -a reference.wav \
    -r "Text spoken in the reference audio" \
    -o my_voice.json

# Use the custom voice
qwen3-tts speak "New text to synthesize" --voice my_voice.json -o output.wav
```

Voice file format (`.json`):
```json
{
  "ref_text": "Text spoken in the reference audio",
  "codec_tokens": [[...], ...]
}
```

> `ref_text` enables In-Context Learning (ICL), aligning reference audio with text for better cloning quality.
> Legacy `.npy` and `.pt` files are also supported (no ref_text, lower quality).

Voice encoding uses a native Candle (Rust ML) implementation of the Mimi Speech Tokenizer —
processes ~4s of audio in ~2s, entirely on CPU.

## Deployment Examples

### Single Machine (all three workers on IP1)

```bash
qwen3-tts init --talker-ip 127.0.0.1 --predictor-ip 127.0.0.1 --vocoder-ip 127.0.0.1
```

```bash
# Terminal 1: Talker
qwen3-tts worker -r talker -b 0.0.0.0:9090

# Terminal 2: Predictor
qwen3-tts worker -r predictor -b 0.0.0.0:9091

# Terminal 3: Vocoder
qwen3-tts worker -r vocoder -b 0.0.0.0:9092

# Terminal 4: Synthesize
qwen3-tts "你好世界"
```

### Two Machines (IP1 = Talker, IP2 = Predictor + Vocoder)

```bash
qwen3-tts init --talker-ip 127.0.0.1 --predictor-ip <IP2> --vocoder-ip <IP2>
```

```bash
# IP1:
qwen3-tts worker -r talker -b 0.0.0.0:9090

# IP2:
qwen3-tts worker -r predictor -b 0.0.0.0:9091
qwen3-tts worker -r vocoder -b 0.0.0.0:9092

# IP1:
qwen3-tts "你好世界"
```

### Three Machines (IP1 = Talker, IP2 = Predictor, IP3 = Vocoder)

```bash
qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP3>
```

```bash
# IP1:
qwen3-tts worker -r talker -b 0.0.0.0:9090

# IP2:
qwen3-tts worker -r predictor -b 0.0.0.0:9091

# IP3:
qwen3-tts worker -r vocoder -b 0.0.0.0:9092

# Any machine (including IP1):
qwen3-tts "你好世界"
```

### Using Deployed Workers from Any Machine

Any machine with the `qwen3-tts` binary and correct config can synthesize speech. The client needs no GPU, NPU, or special hardware.

```bash
cat > qwen3-tts.toml << EOF
[workers.talker]
host = "<IP1>"
port = 9090

[workers.predictor]
host = "<IP2>"
port = 9091

[workers.vocoder]
host = "<IP2>"
port = 9092

[defaults]
language = "chinese"
max_tokens = 200
temperature = 0.8
cp_temperature = 0.1
repetition_penalty = 1.2

[server]
host = "0.0.0.0"
port = 8080
EOF

qwen3-tts "你好世界"
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `qwen3-tts "text"` | Quick speech synthesis (outputs output.wav) |
| `qwen3-tts speak "text" -o file.wav` | Specify output file |
| `qwen3-tts speak "text" --lang english` | Specify language |
| `qwen3-tts speak "text" --voice voice.json` | Voice cloning |
| `qwen3-tts encode-voice -a ref.wav -r "text" -o voice.json` | Create voice profile (native ARM64) |
| `qwen3-tts serve --port 8080` | Start OpenAI-compatible API server |
| `qwen3-tts mcp` | Start MCP server (stdio) |
| `qwen3-tts worker -r talker` | Start Talker Worker |
| `qwen3-tts worker -r predictor` | Start Predictor Worker |
| `qwen3-tts worker -r vocoder` | Start Vocoder Worker |
| `qwen3-tts init --predictor-ip <IP>` | Generate config file |

## OpenAI-Compatible API

```bash
# Start server
qwen3-tts serve --port 8080
```

```bash
# Basic synthesis
curl -X POST http://<IP1>:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "default"}' \
  --output speech.wav

# Voice cloning (voice = path to voice file, supports .json/.npy/.pt)
curl -X POST http://<IP1>:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "/path/to/my_voice.json"}' \
  --output speech.wav
```

Supported parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice file path (`.json`/`.npy`/`.pt`) |
| `language` | string | `"chinese"` | Language |
| `model` | string | `"qwen3-tts"` | Model name |
| `response_format` | string | `"wav"` | Output format |

## MCP Server

Provides AI tools via stdio JSON-RPC:

```bash
qwen3-tts mcp
```

### Tools

- **text_to_speech** — Text-to-speech with voice cloning support

### Configuration (Claude Desktop / Cursor etc.)

```json
{
  "mcpServers": {
    "qwen3-tts": {
      "command": "/path/to/qwen3-tts",
      "args": ["mcp"]
    }
  }
}
```

### Example Call

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "text_to_speech",
    "arguments": {
      "text": "Hello, this is a voice cloning test.",
      "voice": "/path/to/my_voice.json",
      "output_path": "output.wav"
    }
  }
}
```

## Configuration

Path priority: `./qwen3-tts.toml` > `~/.config/qwen3-tts/config.toml`

```toml
[workers.talker]
host = "127.0.0.1"       # Talker Worker IP
port = 9090

[workers.predictor]
host = "<YOUR_IP>"        # ⚠️ Change to your Predictor IP
port = 9091

[workers.vocoder]
host = "<YOUR_IP>"        # ⚠️ Change to your Vocoder IP
port = 9092

[defaults]
language = "chinese"
max_tokens = 200
temperature = 0.8
cp_temperature = 0.1
repetition_penalty = 1.2

# EOS convergence parameters (tunable, defaults work for most cases)
# eos_start_ratio = 0.6     # Start boosting EOS at 60% of estimated tokens
# eos_max_ratio = 1.2       # Max boost at 120%
# eos_force_ratio = 1.5     # Force stop at 150%
# eos_max_boost = 25.0      # Maximum EOS logit increment

[server]
host = "0.0.0.0"
port = 8080
```

## HuggingFace Model Structure

```
kautism/qwen3-tts-rk3588/
├── talker/                        # Talker Worker
│   ├── talker-q8_0.gguf          (768 MB, Q8 LLM)
│   ├── tokenizer.json            (11 MB)
│   └── embeddings/               (~1.2 GB)
├── predictor/                     # Predictor Worker
│   ├── code_predictor/
│   ├── qwen3-tts-0.6b-q8_0.gguf  (1.3 GB, Candle GGUF Q8_0)
│   │   └── config.json
│   └── embeddings/
├── vocoder/                       # Vocoder Worker
│   ├── vocoder.onnx              (436 MB, FP32 CPU — default)
│   └── vocoder.rknn              (128 MB, INT8 NPU — needs rknn-vocoder feature)
└── speech_tokenizer/              # Voice encoding (encode-voice)
    └── model.safetensors         (651 MB, Mimi encoder)
```

Workers auto-download their role's models from HuggingFace Hub on first start.

## Supported Languages

Chinese · English · Deutsch · Русский · Français · 日本語 · 한국어

## Performance

| Metric | Candle (default) | GGML (`--features ggml-backend`) |
|--------|-----------------|----------------------------------|
| Token generation rate | ~3.6 tok/s | **~4.0 tok/s** |
| Talker latency | ~60ms/step | ~33ms/step |
| Predictor latency | ~185ms/step | ~185ms/step |
| Vocoder (ONNX FP32) | ~4.5s (CPU, clean audio) | ~4.5s |
| Vocoder (RKNN INT8) | ~2.7s (NPU, quantization noise) | ~2.7s |
| RTF — ONNX (default) | ~5.0x | ~3.8x |
| RTF — RKNN | ~4.2x | ~3.5x |
| Voice encoding speed | ~2s/4s audio | ~2s/4s audio |
| Network overhead | <5ms/step (LAN) | <5ms/step (LAN) |
| External dependencies | `libonnxruntime.so` | Multiple `.so` (see table above) |

> RTF = generation time / audio duration. RTF < 1 is real-time.
> Candle backend includes SDOT inline assembly and pre-allocated memory pool optimizations.

## Support the Project

If this project has saved you time or helped you in your workflow, consider supporting its continued development.

[![][ko-fi-shield]][ko-fi-link]
[![][paypal-shield]][paypal-link]

<!-- Link Definitions -->

[license-shield]: https://img.shields.io/badge/license-MIT-white?labelColor=black&style=flat-square
[license-shield-link]: https://github.com/darkautism/qwen3-tts/blob/main/LICENSE
[last-commit-shield]: https://img.shields.io/github/last-commit/darkautism/qwen3-tts?color=c4f042&labelColor=black&style=flat-square
[last-commit-shield-link]: https://github.com/darkautism/qwen3-tts/commits/main
[ko-fi-shield]: https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white
[ko-fi-link]: https://ko-fi.com/kautism
[paypal-shield]: https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white
[paypal-link]: https://paypal.me/kautism
