<div align="center">

![banner](doc/images/banner.png)

# qwen3-tts

Distributed Qwen3-TTS — Rust-based distributed text-to-speech optimized for low-cost SBC clusters.

We have a Web UI now! Start the server and open the built-in web interface in your browser.

Single binary provides CLI, an OpenAI-compatible HTTP API (with web UI), an MCP stdio bridge, and inference workers. Models auto-download from HuggingFace Hub.


[![][license-shield]][license-shield-link]
[![][last-commit-shield]][last-commit-shield-link]


[中文文檔](README-zh.md)

We have WebUI Now!

![webui](doc/images/webui.jpg)

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

# IP2 - Predictor Worker (Q4 quantization for max speed)
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores --quant q4

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

## OpenAI-Compatible API & Web UI

```bash
# Start server (includes built-in web UI at http://localhost:8080)
qwen3-tts serve --port 8080
```

### Web UI

Open `http://<server-ip>:8080` in a browser. The built-in web UI provides:

- **Text input** with Ctrl+Enter to synthesize
- **Voice selector** dropdown — populated from 500+ voices on [HuggingFace](https://huggingface.co/kautism/qwen3_tts_voices_json), grouped by game/character
- **Language selector** (Chinese, English, Japanese, Korean)
- **Audio playback** directly in browser

No installation required — pure HTML/JS, served by the same binary.

### REST API

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
│   │   ├── code-predictor-q8_0.gguf  (206 MB, stripped — default download)
│   │   ├── code-predictor-q4_0.gguf  (169 MB, stripped Q4 — use --quant q4)
│   │   └── qwen3-tts-0.6b-q8_0.gguf (1.3 GB, full model — not recommended)
│   └── embeddings/
├── vocoder/                       # Vocoder Worker
│   ├── vocoder.onnx              (436 MB, FP32 CPU — default)
│   └── vocoder.rknn              (128 MB, INT8 NPU — needs rknn-vocoder feature)
└── speech_tokenizer/              # Voice encoding (encode-voice)
    └── model.safetensors         (651 MB, Mimi encoder)
```

Workers auto-download their role's models from HuggingFace Hub on first start.
Pass `--quant q4` to predictor workers to also download the Q4 model (169MB, ~16% faster).

## Supported Languages

Chinese · English · Deutsch · Русский · Français · 日本語 · 한국어

## Speed Optimization Guide

To achieve the best RTF on ARM SBCs, apply these optimizations (in order of impact):

### 1. Use `--big-cores` (essential on big.LITTLE SoCs)

```bash
# Pins all threads to performance cores (e.g., A76 on RK3588)
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores
```

Without this, rayon distributes matmul work to slow efficiency cores → **~43% slower**.

### 2. Use Q4 quantization (`--quant q4`)

```bash
# Q4 model is 169MB vs 206MB for Q8. 16% faster prediction, comparable quality.
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores --quant q4
```

The Q4 GGUF is auto-downloaded from HuggingFace Hub when `--quant q4` is specified.

### 3. Distribute workers across machines

```bash
# Machine 1: talker only (gets all 4 big cores to itself)
qwen3-tts worker -r talker -b 0.0.0.0:9090 --big-cores

# Machine 2: predictor + vocoder (their own 4 big cores)
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --big-cores --quant q4
qwen3-tts worker -r vocoder -b 0.0.0.0:9092 --big-cores
```

Edit `qwen3-tts.toml` to point to remote workers:
```toml
[workers.talker]
host = "192.168.1.10"   # Machine 1
port = 9090

[workers.predictor]
host = "192.168.1.11"   # Machine 2
port = 9091

[workers.vocoder]
host = "192.168.1.11"   # Machine 2
port = 9092
```

This eliminates core contention and gives ~10% RTF improvement.

### 4. Build with SDOT (ARM dotprod)

```bash
# Enable ARM SDOT instruction for faster quantized matmul
RUSTFLAGS='-C target-feature=+dotprod' cargo build --release
```

This is critical on Cortex-A76 and newer cores. Without it, quantized inference uses a slower vmull+vpaddl path.

### Summary: Cumulative Effect| What | MEDIUM RTF | Speedup |
|------|-----------|---------|
| Default (no optimization) | 4.96× | baseline |
| + `--big-cores` + Q4 + SDOT | 2.87× | **42% faster** |
| + distribute to 2 machines | **2.61×** | **47% faster** |

## Performance

Tested on RK3588 (4×A76 + 4×A55), workers started with `--big-cores --quant q4`:

### Single Machine vs Two Machines

| Test | 1× RK3588 | 2× RK3588 (LAN) | Improvement |
|------|-----------|------------------|-------------|
| SHORT (~40 tokens) | 3.30× | **2.99×** | 9% |
| MEDIUM (~175 tokens) | 2.87× | **2.61×** | 9% |
| LONG (200 tokens) | 3.31× | **2.99×** | 10% |

The second machine offloads predictor + vocoder, freeing all 4 A76 cores on machine 1 for the talker.
Single-machine works well for simple deployments; adding a second board reduces RTF by ~10%.

| Metric | 1 Machine | 2 Machines |
|--------|-----------|------------|
| Token rate | ~4.9 tok/s | **~5.5 tok/s** |
| Talker latency | ~35ms/step | ~35ms/step |
| Predictor latency | ~93ms/step | ~93ms/step |
| Vocoder (ONNX FP32) | ~4.5s/batch | ~4.5s/batch |
| Vocoder (RKNN INT8) | ~2.7s/batch (⚠️ has noise) | ~2.7s/batch |
| **Best MEDIUM RTF** | 2.87× | **2.61×** |
| External dependencies | `libonnxruntime.so` only | same |

> RTF = generation time / audio duration. Lower is better; RTF < 1 is real-time.
>
> **Important:** On big.LITTLE SoCs (e.g., RK3588), always start workers with `--big-cores` to avoid slow small cores.
> Without it, rayon distributes matmul to slow A55 cores → predictor degrades from 93ms to 190ms.

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
