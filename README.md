<div align="center">

# qwen3-tts

Distributed Qwen3-TTS speech synthesis system — a Rust implementation designed for low-cost SBC clusters.

Single binary serves as CLI, OpenAI-compatible API server, MCP server, and inference worker.
Models auto-download from HuggingFace Hub.
**Zero Python dependencies** — all inference uses Candle (Rust ML framework), voice encoding is native.
Single statically-compiled binary; Talker/Predictor require no external `.so` libraries.

[![][license-shield]][license-shield-link]
[![][last-commit-shield]][last-commit-shield-link]

[中文文檔](README-zh.md)

</div>

## Architecture

```
 Machine 1 (SBC / Server)              Machine 2 (SBC / Server)
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
| **Talker** | Tokenizer + TextEmbedder + LLM | CPU | 9090 |
| **Predictor** | CodePredictor (Candle GGUF Q8_0) + feedback embedding | CPU | 9091 |
| **Vocoder** | Vocoder (ONNX FP32 CPU) | CPU | 9092 |

Distributes workload across low-cost SBCs or any Linux machines. Token generation: ~5.5 tok/s with stripped code-predictor GGUF (default).

## Requirements

### Hardware
- 1–3 Linux machines (low-cost ARM SBCs work well; 4GB+ RAM per node)
- Any aarch64 or x86_64 Linux system — tested on RK3588, should work on other ARM boards

### Runtime Dependencies

Inference core (Talker, Predictor) uses Candle (pure Rust) — **no external libraries needed**.

**Vocoder machine requires:**

| Library | Source | Install Path |
|---------|--------|-------------|
| `libonnxruntime.so` | `pip install onnxruntime` (auto-detected) | Python package or system path |

### Installing ONNX Runtime (Vocoder Machine)

```bash
# Easiest: via Python package (auto-detected by the binary)
pip install onnxruntime

# Or manual system install
# Download aarch64 build: https://github.com/microsoft/onnxruntime/releases
sudo cp libonnxruntime.so /usr/lib/
```

### (Optional) RKNN NPU Acceleration

With `--features rknn-vocoder`, the vocoder runs on Rockchip NPU instead of CPU.
**Note:** RKNN INT8 quantization introduces audible noise in the output. Use only if RTF reduction is critical.

```bash
# Requires librknnrt.so and RKNPU kernel driver
sudo curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so \
  -o /lib/librknnrt.so
```

## Building

```bash
# Standard build (Candle inference + ONNX FP32 vocoder — pure Rust, no extra .so)
cargo build --release

# RKNN INT8 vocoder (Rockchip NPU only — faster but introduces quantization noise)
cargo build --release --features rknn-vocoder
# Output: target/release/qwen3-tts (~15-20 MB)
```

> Cross-compile: `cross build --release --target aarch64-unknown-linux-gnu`

### Feature Gates

| Feature | Description | Extra Dependencies | Performance |
|---------|-------------|-------------------|-------------|
| (default) | Candle inference + ONNX vocoder | `libonnxruntime.so` | **~5.5 tok/s** |
| `ggml-predictor` | C++ GGML code predictor | Static `.a` libs | Slower than Candle |
| `rknn-vocoder` | RKNN INT8 vocoder (Rockchip NPU) | `librknnrt.so` + RKNPU kernel | ⚠️ has noise |

Default uses Candle (pure Rust) inference — no C/C++ library installation needed.
The Candle backend includes SDOT inline assembly optimization and benefits greatly from stripped GGUF models.
RKNN vocoder trades audio quality for speed — INT8 quantization introduces audible artifacts.

## Quick Start

### Initialize Configuration

```bash
# Generate qwen3-tts.toml with your worker IPs
qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP2>
```

### Start Workers (models auto-download from HF Hub)

```bash
# IP1 - Talker Worker (use --big-cores on big.LITTLE SoCs like RK3588)
qwen3-tts worker -r talker -b 0.0.0.0:9090 --big-cores

# IP2 - Predictor Worker
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores

# IP2 - Vocoder Worker (can share machine with Predictor)
qwen3-tts worker -r vocoder -b 0.0.0.0:9092 --big-cores
```

> Models download to `~/.local/share/qwen3-tts/models/{role}/` by default.
> Custom HF repo: `--repo your-name/your-repo`
> For specific core pinning: `--cores 4-7` or `--cores 4,5,6,7`

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

Runs directly on ARM SBCs — **no x86 or Python needed**:

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
qwen3-tts worker -r talker -b 0.0.0.0:9090 --big-cores

# Terminal 2: Predictor
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores

# Terminal 3: Vocoder
qwen3-tts worker -r vocoder -b 0.0.0.0:9092 --big-cores

# Terminal 4: Synthesize
qwen3-tts "你好世界"
```

```bash
qwen3-tts init --talker-ip 127.0.0.1 --predictor-ip <IP2> --vocoder-ip <IP2>
```

```bash
# IP1:
qwen3-tts worker -r talker -b 0.0.0.0:9090 --big-cores

# IP2:
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores
qwen3-tts worker -r vocoder -b 0.0.0.0:9092 --big-cores

# IP1:
qwen3-tts "你好世界"
```

### Three Machines (IP1 = Talker, IP2 = Predictor, IP3 = Vocoder)

```bash
qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP3>
```

```bash
# IP1:
qwen3-tts worker -r talker -b 0.0.0.0:9090 --big-cores

# IP2:
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores

# IP3:
qwen3-tts worker -r vocoder -b 0.0.0.0:9092 --big-cores

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
| `qwen3-tts worker -r talker --big-cores` | Worker pinned to big CPU cores |
| `qwen3-tts worker -r talker --cores 4-7` | Worker pinned to specific cores |
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
│   │   ├── code-predictor-q8_0.gguf  (206 MB, stripped — recommended)
│   │   └── qwen3-tts-0.6b-q8_0.gguf (1.3 GB, full model)
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

Tested on 2× RK3588 (4×A76+4×A55) over Gigabit LAN, workers started with `--big-cores`:

| Metric | Value |
|--------|-------|
| Token generation rate | **~5.5 tok/s** |
| Talker latency | ~40ms/step |
| Predictor latency | **~108ms/step** |
| Vocoder (ONNX FP32) | ~4.5s (CPU, clean audio) |
| Vocoder (RKNN INT8) | ~2.7s (NPU, ⚠️ has noise) |
| **RTF (2 machines)** | **~3.1x** |
| Voice encoding speed | ~2s/4s audio |
| Network overhead | <5ms/step (LAN) |
| External dependencies | `libonnxruntime.so` only |

> RTF = generation time / audio duration. Lower is better; RTF < 1 is real-time.
>
> **Important:** On big.LITTLE SoCs (e.g., RK3588), always start workers with `--big-cores` to avoid slow small cores.
> Without it, rayon distributes matmul to slow A55 cores → predictor degrades from 108ms to 190ms.

### Optimization Journey

| Optimization | Predictor (ms/step) | MEDIUM RTF | Change |
|---|---|---|---|
| Baseline (Candle Q8_0, full 1.3GB GGUF) | 185 | 4.96× | — |
| + Server-side past_tokens + mem::take() | 185 | 4.79× | −3% |
| + **Stripped code-predictor GGUF (206MB)** | **108** | **3.12×** | **−37%** |
| + **CPU affinity (`--big-cores`)** | **108** | **3.08×** | **−1%** |
| + Q4_0 quantization (`--quant q4`) | 93 | ~2.80× | −3% |
| + **Typed wire protocol** (ResponseData enum) | 93 | **2.58×** | **−8%** |

Key insight: the full 1.3GB GGUF contains talker weights (1075MB) never used by the predictor.
Stripping to a 206MB code-predictor-only GGUF eliminates L2 cache pollution → **41% faster prediction**.

The `scripts/extract_code_predictor_gguf.py` tool creates stripped GGUFs from full models.

> **Q4 vs Q8:** Q4 gives ~16% faster prediction (93ms vs 107ms) with comparable quality.
> Use `--quant q4` on the predictor worker for speed, `--quant q8` (default) for safety.

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
