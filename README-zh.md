<div align="center">

# qwen3-tts

分散式 Qwen3-TTS 語音合成系統，專為低成本 SBC 叢集設計的 Rust 實作。

單一二進位檔同時作為 CLI、OpenAI API 伺服器、MCP 伺服器與推理 Worker。
模型自動從 HuggingFace Hub 下載。
**零 Python 依賴** — 全部推理使用 Candle (Rust ML 框架)，語音編碼亦原生執行。
單一靜態編譯二進位，Talker/Predictor 不需要任何外部 .so 函式庫。

[![][license-shield]][license-shield-link]
[![][last-commit-shield]][last-commit-shield-link]

</div>

## 架構

```
 Machine 1 (SBC / 伺服器)              Machine 2 (SBC / 伺服器)
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

**三個 Worker 角色：**

| Worker | 功能 | 運算資源 | 預設 Port |
|--------|------|----------|-----------|
| **Talker** | Tokenizer + TextEmbedder + LLM | CPU | 9090 |
| **Predictor** | CodePredictor (Candle GGUF Q8_0) + 回饋嵌入 | CPU | 9091 |
| **Vocoder** | Vocoder (ONNX FP32 CPU) | CPU | 9092 |

將工作分散在低成本 SBC 或任何 Linux 機器上。Token 生成：Candle ~3.6 tok/s (預設) / GGML ~4.0 tok/s (`--features ggml-backend`)。

## 系統需求

### 硬體
- 1～3 台 Linux 機器（低成本 ARM SBC 即可；每節點 4GB+ RAM）
- 任何 aarch64 或 x86_64 Linux 系統 — 已在 RK3588 測試，其他 ARM 開發板亦可

### 執行期依賴

推理核心 (Talker、Predictor) 使用 Candle (純 Rust)，**無需安裝任何外部函式庫**。

**Vocoder 機器需要：**

| 函式庫 | 來源 | 安裝路徑 |
|--------|------|----------|
| `libonnxruntime.so` | `pip install onnxruntime` (自動偵測) | Python 套件或系統路徑 |

### 安裝 ONNX Runtime (Vocoder 機器)

```bash
# 最簡方式：透過 Python 套件 (程式會自動偵測)
pip install onnxruntime

# 或手動安裝到系統路徑
# 下載 aarch64 版本: https://github.com/microsoft/onnxruntime/releases
sudo cp libonnxruntime.so /usr/lib/
```

### (選用) RKNN NPU 加速

使用 `--features rknn-vocoder` 時，vocoder 會在 Rockchip NPU 上執行。
**注意：** RKNN INT8 量化會在輸出中引入可聽見的雜音。僅在 RTF 至關重要時使用。

```bash
# 需要 librknnrt.so 和 RKNPU kernel driver
sudo curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so \
  -o /lib/librknnrt.so
```

## 編譯

```bash
# 標準編譯 (Candle 推理 + ONNX FP32 vocoder — 純 Rust，無額外 .so)
cargo build --release

# 使用 C++ GGML 後端 (ARM NEON SDOT 加速，~2x 快於 Candle，需 llama_wrapper.so)
cargo build --release --features ggml-backend

# 使用 RKNN INT8 vocoder (僅 Rockchip NPU — 較快但有量化雜音)
cargo build --release --features rknn-vocoder

# 兩者都啟用
cargo build --release --features ggml-backend,rknn-vocoder
# 產出: target/release/qwen3-tts (~15-20 MB)
```

> 交叉編譯可使用 `cross build --release --target aarch64-unknown-linux-gnu`

### Feature Gates

| Feature | 說明 | 額外依賴 | 效能 |
|---------|------|----------|------|
| (預設) | Candle 推理 + ONNX vocoder | `libonnxruntime.so` | ~3.6 tok/s |
| `ggml-backend` | C++ GGML/llama.cpp 推理 | `llama_wrapper.so` + `libllama.so` + `libggml*.so` | **~4.0 tok/s** |
| `rknn-vocoder` | RKNN INT8 vocoder (Rockchip NPU) | `librknnrt.so` + RKNPU kernel | ⚠️ 有雜音 |

預設使用 Candle (純 Rust) 推理，不需額外安裝 C/C++ 函式庫。
啟用 `ggml-backend` 可利用 ARM NEON SDOT 硬體指令額外加速約 10-15%。
Candle 後端已包含 SDOT 內聯組語優化 + 預分配記憶體池，差距已大幅縮小。
RKNN vocoder 以音質換取速度 — INT8 量化會引入可聽見的雜音。

## 快速開始

### 初始化配置

```bash
# 產生 qwen3-tts.toml，設定你的 Worker IP
qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP2>
```

### 啟動 Worker (模型自動從 HF Hub 下載)

```bash
# IP1 - Talker Worker
qwen3-tts worker -r talker -b 0.0.0.0:9090

# IP2 - Predictor Worker
qwen3-tts worker -r predictor -b 0.0.0.0:9091

# IP2 - Vocoder Worker (可以和 Predictor 同一台)
qwen3-tts worker -r vocoder -b 0.0.0.0:9092
```

> 模型預設下載到 `~/.local/share/qwen3-tts/models/{role}/`
> 自定義 HF repo: `--repo your-name/your-repo`

### 合成語音

```bash
# 簡單用法 (輸出 output.wav)
qwen3-tts "你好世界"

# 指定輸出和語言
qwen3-tts speak "Hello world" -o speech.wav --lang english

# 聲音克隆 (自訂聲音檔)
qwen3-tts speak "你好" --voice my_voice.json -o clone.wav
```

## 聲音克隆

### 製作聲音

直接在 ARM SBC 上運行，**不需要 x86 或 Python**：

```bash
# 編碼參考音訊 → 輸出單一 .json 聲音檔 (支援任意取樣率，自動重取樣至 24kHz)
qwen3-tts encode-voice \
    -a reference.wav \
    -r "參考音檔中說的文字內容" \
    -o my_voice.json

# 使用自訂聲音
qwen3-tts speak "要合成的新文字" --voice my_voice.json -o output.wav
```

聲音檔格式 (`.json`)：
```json
{
  "ref_text": "參考音檔中說的文字內容",
  "codec_tokens": [[...], ...]
}
```

> `ref_text` 用於 In-Context Learning (ICL)，讓模型對齊參考音訊與文字，提升克隆品質。
> 也支援舊格式 `.npy` 和 `.pt` 檔案 (無 ref_text，品質較低)。

語音編碼使用原生 Candle (Rust ML 框架) 實作的 Mimi Speech Tokenizer，
約 2 秒即可處理 4 秒音訊，完全在 CPU 上運行。

## 部署範例

### 一台機器 (IP1 跑全部三個 Worker)

```bash
qwen3-tts init --talker-ip 127.0.0.1 --predictor-ip 127.0.0.1 --vocoder-ip 127.0.0.1
```

```bash
# 終端 1: Talker
qwen3-tts worker -r talker -b 0.0.0.0:9090

# 終端 2: Predictor
qwen3-tts worker -r predictor -b 0.0.0.0:9091

# 終端 3: Vocoder
qwen3-tts worker -r vocoder -b 0.0.0.0:9092

# 終端 4: 合成
qwen3-tts "你好世界"
```

### 兩台機器 (IP1 = Talker, IP2 = Predictor + Vocoder)

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

### 三台機器 (IP1 = Talker, IP2 = Predictor, IP3 = Vocoder)

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

# 任意機器 (含 IP1):
qwen3-tts "你好世界"
```

### 從任意機器使用已部署的 Worker

只要有 `qwen3-tts` 二進位和正確的配置檔，**任何能連上 Worker 的機器**都可以做語音合成。控制端不需要 GPU、NPU 或特殊硬體。

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

## CLI 參考

| 指令 | 說明 |
|------|------|
| `qwen3-tts "文字"` | 快速語音合成 (輸出 output.wav) |
| `qwen3-tts speak "文字" -o file.wav` | 指定輸出檔案 |
| `qwen3-tts speak "文字" --lang english` | 指定語言 |
| `qwen3-tts speak "文字" --voice voice.json` | 聲音克隆 |
| `qwen3-tts encode-voice -a ref.wav -r "文字" -o voice.json` | 製作聲音檔 (ARM64 原生) |
| `qwen3-tts serve --port 8080` | 啟動 OpenAI 相容 API |
| `qwen3-tts mcp` | 啟動 MCP 伺服器 (stdio) |
| `qwen3-tts worker -r talker` | 啟動 Talker Worker |
| `qwen3-tts worker -r predictor` | 啟動 Predictor Worker |
| `qwen3-tts worker -r vocoder` | 啟動 Vocoder Worker |
| `qwen3-tts init --predictor-ip <IP>` | 產生配置檔 |

## OpenAI 相容 API

```bash
# 啟動伺服器
qwen3-tts serve --port 8080
```

```bash
# 基本合成
curl -X POST http://<IP1>:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "default"}' \
  --output speech.wav

# 聲音克隆 (voice 填聲音檔的路徑，支援 .json/.npy/.pt)
curl -X POST http://<IP1>:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "/path/to/my_voice.json"}' \
  --output speech.wav
```

支援參數：

| 參數 | 類型 | 預設 | 說明 |
|------|------|------|------|
| `input` | string | (必填) | 要合成的文字 |
| `voice` | string | `"default"` | 聲音檔路徑 (`.json`/`.npy`/`.pt`) |
| `language` | string | `"chinese"` | 語言 |
| `model` | string | `"qwen3-tts"` | 模型名稱 |
| `response_format` | string | `"wav"` | 輸出格式 |

## MCP 伺服器

透過 stdio JSON-RPC 提供 AI 工具：

```bash
qwen3-tts mcp
```

### 工具

- **text_to_speech** — 文字轉語音，支援聲音克隆

### 設定 (Claude Desktop / Cursor 等)

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

### 範例呼叫

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "text_to_speech",
    "arguments": {
      "text": "你好，這是聲音克隆測試。",
      "voice": "/path/to/my_voice.json",
      "output_path": "output.wav"
    }
  }
}
```

## 設定檔

路徑優先順序：`./qwen3-tts.toml` > `~/.config/qwen3-tts/config.toml`

```toml
[workers.talker]
host = "127.0.0.1"       # Talker Worker IP
port = 9090

[workers.predictor]
host = "<YOUR_IP>"        # ⚠️ 改成你的 Predictor IP
port = 9091

[workers.vocoder]
host = "<YOUR_IP>"        # ⚠️ 改成你的 Vocoder IP
port = 9092

[defaults]
language = "chinese"
max_tokens = 200
temperature = 0.8
cp_temperature = 0.1
repetition_penalty = 1.2

# EOS 收斂參數 (可微調，預設值適用大多數情況)
# eos_start_ratio = 0.6     # 從預估 token 數的 60% 開始加 boost
# eos_max_ratio = 1.2       # 在 120% 時 boost 達到最大值
# eos_force_ratio = 1.5     # 在 150% 時強制停止
# eos_max_boost = 25.0      # 最大 EOS logit 增量

[server]
host = "0.0.0.0"
port = 8080
```

## HuggingFace 模型結構

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
│   ├── vocoder.onnx              (436 MB, FP32 CPU — 預設)
│   └── vocoder.rknn              (128 MB, INT8 NPU — 需 rknn-vocoder feature)
└── speech_tokenizer/              # 語音編碼 (encode-voice)
    └── model.safetensors         (651 MB, Mimi encoder)
```

Worker 啟動時會自動從 HuggingFace Hub 下載對應角色的模型。

## 支援語言

中文 · English · Deutsch · Русский · Français · 日本語 · 한국어

## 效能

| 指標 | Candle (預設) | GGML (`--features ggml-backend`) |
|------|--------------|----------------------------------|
| Token 生成速率 | ~3.6 tok/s | **~4.0 tok/s** |
| Talker 延遲 | ~60ms/step | ~33ms/step |
| Predictor 延遲 | ~185ms/step | ~185ms/step |
| Vocoder (ONNX FP32) | ~4.5s (CPU, 無雜音) | ~4.5s |
| Vocoder (RKNN INT8) | ~2.7s (NPU, ⚠️ 有雜音) | ~2.7s |
| RTF — ONNX (預設) | ~4.8x | ~3.8x |
| RTF — RKNN | ~4.2x | ~3.5x |
| 語音編碼速度 | ~2s/4s audio | ~2s/4s audio |
| 網路開銷 | <5ms/step (LAN) | <5ms/step (LAN) |
| 外部依賴 | `libonnxruntime.so` | `.so` 多個 (見上表) |

> RTF = 生成時間 / 音訊時間。RTF < 1 為即時。
> Candle 後端已包含 SDOT 內聯組語和預分配記憶體池優化。

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
