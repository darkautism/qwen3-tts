<div align="center">

![banner](doc/images/banner.png)

# qwen3-tts

分散式 Qwen3-TTS 語音合成系統，專為低成本 SBC 叢集設計的 Rust 實作。

單一二進位檔同時作為 CLI、OpenAI API 伺服器、MCP 伺服器與推理 Worker。
模型自動從 HuggingFace Hub 下載。
**零 Python 依賴** — 全部推理使用 Candle (Rust ML 框架)，語音編碼亦原生執行。
單一靜態編譯二進位，Talker/Predictor 不需要任何外部 .so 函式庫。

[![][license-shield]][license-shield-link]
[![][last-commit-shield]][last-commit-shield-link]


We have WebUI Now!

![webui](doc/images/webui.jpg)

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

將工作分散在低成本 SBC 或任何 Linux 機器上。Token 生成：~5.5 tok/s（使用精簡版 code-predictor GGUF）。

## 系統需求

### 硬體
- 1～3 台 Linux 機器（低成本 ARM SBC 即可；每節點 4GB+ RAM）
- 任何 aarch64 或 x86_64 Linux 系統 — 已在 RK3588 測試，其他 ARM 開發板亦可

### 執行期依賴

推理核心 (Talker、Predictor) 使用 Candle (純 Rust)，**無需安裝任何外部函式庫**。

**Vocoder 機器需要：**

| 函式庫 | 來源 | 安裝路徑 |
|--------|------|----------|
| `libonnxruntime.so` | `uv pip install onnxruntime` 或 `uv pip install onnxruntime-gpu` | 系統路徑或 `ORT_DYLIB_PATH` |

### 安裝 ONNX Runtime (Vocoder 機器)

```bash
# CPU (uv)
uv pip install onnxruntime

# CUDA (uv, GPU 版)
uv pip install onnxruntime-gpu

# uvx 替代方案
uvx pip install onnxruntime
uvx pip install onnxruntime-gpu

# 若未在系統路徑內，請設定 ORT_DYLIB_PATH
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

> 使用 `--features onnx-cuda` 時，**CUDA Toolkit 與 cuDNN 都要安裝**，缺一個都會導致 CUDA EP 初始化失敗。

#### 從 ONNX Runtime v1.24.2 手動下載

下載頁面：
`https://github.com/microsoft/onnxruntime/releases/tag/v1.24.2`

解壓後，把 `ORT_DYLIB_PATH` 指到動態函式庫：
- **Linux x64 GPU**：`.../onnxruntime-linux-x64-gpu-1.24.2/lib/libonnxruntime.so`
- **Linux aarch64 CPU**：`.../onnxruntime-linux-aarch64-1.24.2/lib/libonnxruntime.so`

也可以把 `libonnxruntime.so` 放到系統動態函式庫搜尋路徑。

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

# CUDA ONNX (vocoder + ONNX predictor 使用 GPU)
cargo build --release --features onnx-cuda

# 使用 RKNN INT8 vocoder (僅 Rockchip NPU — 較快但有量化雜音)
cargo build --release --features rknn-vocoder
# 產出: target/release/qwen3-tts (~15-20 MB)
```

> 交叉編譯可使用 `cross build --release --target aarch64-unknown-linux-gnu`

### Feature Gates

| Feature | 說明 | 額外依賴 | 效能 |
|---------|------|----------|------|
| (預設) | Candle 推理 + ONNX vocoder | `libonnxruntime.so` | **~5.5 tok/s** |
| `onnx-cuda` | ONNX CUDA EP（vocoder + predictor 的 ONNX 回退路徑） | `onnxruntime-gpu` + CUDA/cuDNN | 依 GPU 而定 |
| `ggml-backend` | llama.cpp talker + GGML predictor | `llama_wrapper.so` + 靜態 GGML 庫 | 在調校 ARM 上最快 |
| `ggml-predictor` | 僅 GGML predictor（talker 保持 Candle） | 靜態 GGML 庫 | 僅加速 predictor |
| `rknn-vocoder` | RKNN INT8 vocoder (Rockchip NPU) | `librknnrt.so` + RKNPU kernel | ⚠️ 有雜音 |

預設使用 Candle (純 Rust) 推理，不需額外安裝 C/C++ 函式庫。
Candle 後端包含 SDOT 內聯組語優化，搭配精簡版 GGUF 模型可大幅提升效能。
RKNN vocoder 以音質換取速度 — INT8 量化會引入可聽見的雜音。
Predictor **不需要同時** ONNX + GGUF：會優先走 GGUF（GGML/Candle），只有在 GGUF 缺失時才回退到 ONNX。

## 快速開始

### 初始化配置

```bash
# 產生 ~/.config/qwen3-tts/config.toml，設定你的 Worker IP
qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP2>
```

### 啟動 Worker (模型自動從 HF Hub 下載)

```bash
# IP1 - Talker Worker（RK3588 會預設自動綁定大核心）
qwen3-tts worker -r talker -b 0.0.0.0:9090

# IP2 - Predictor Worker（Q4 量化以獲得最快速度）
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --quant q4

# IP2 - Vocoder Worker (可以和 Predictor 同一台)
qwen3-tts worker -r vocoder -b 0.0.0.0:9092
```

> qwen3-tts 會優先使用 HuggingFace Hub cache 路徑（由 `hf-hub` `repo.get(...)` 回傳）。
> 只有在 Hub 解析失敗且本地 `${XDG_DATA_HOME:-$HOME/.local/share}/qwen3-tts/models/{role}/` 檔案完整時，才回退到本地檔案。
> 自定義 HF repo: `--repo your-name/your-repo`
> 指定 CPU 核心: `--cores 4-7` 或 `--cores 4,5,6,7`
> 在 big.LITTLE SoC（RK3588）上，預設已啟用大核心綁定，不需要額外參數。

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
| `qwen3-tts worker -r talker` | 啟動 Talker Worker（RK3588 預設自動綁定大核心） |
| `qwen3-tts worker -r predictor` | 啟動 Predictor Worker |
| `qwen3-tts worker -r vocoder` | 啟動 Vocoder Worker |
| `qwen3-tts worker -r talker --big-cores` | 舊版相容參數（現已是預設行為） |
| `qwen3-tts worker -r talker --cores 4-7` | Worker 固定在指定核心上 |
| `qwen3-tts init --predictor-ip <IP>` | 產生配置檔 |

## OpenAI 相容 API 與 Web UI

```bash
# 啟動伺服器（內建 Web UI 於 http://localhost:8080）
qwen3-tts serve --port 8080
```

### Web UI

在瀏覽器開啟 `http://<伺服器-ip>:8080`。內建 Web UI 提供：

- **文字輸入框** — Ctrl+Enter 合成語音
- **聲音選擇器** — 從 [HuggingFace](https://huggingface.co/kautism/qwen3_tts_voices_json) 載入 500+ 聲音，依遊戲/角色分組
- **語言選擇** （中文、英文、日文、韓文）
- **直接在瀏覽器播放音訊**

無需安裝任何東西 — 純 HTML/JS，由同一個二進位檔提供。

### REST API

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
│   │   ├── code-predictor-q8_0.gguf  (206 MB, 精簡版 — 預設下載)
│   │   ├── code-predictor-q4_0.gguf  (169 MB, 精簡 Q4 — 搭配 --quant q4)
│   │   └── qwen3-tts-0.6b-q8_0.gguf (1.3 GB, 完整模型 — 不建議)
│   └── embeddings/
├── vocoder/                       # Vocoder Worker
│   ├── vocoder.onnx              (436 MB, FP32 CPU — 預設)
│   └── vocoder.rknn              (128 MB, INT8 NPU — 需 rknn-vocoder feature)
└── speech_tokenizer/              # 語音編碼 (encode-voice)
    └── model.safetensors         (651 MB, Mimi encoder)
```

Worker 啟動時會自動從 HuggingFace Hub 下載對應角色的模型。
Predictor 使用 `--quant q4` 時只會下載/使用 Q4 predictor GGUF (169MB，快 ~16%)。

## 支援語言

中文 · English · Deutsch · Русский · Français · 日本語 · 한국어

## 加速指南

以下優化按影響力排序，逐步套用可將 RTF 從 4.96× 降至 2.61×：

### 1. 自動大核心綁定（big.LITTLE SoC 預設）

```bash
# 現在預設會自動綁定到效能核心（如 RK3588 的 A76）
qwen3-tts worker -r predictor -b 0.0.0.0:9091
```

若沒有大核心綁定，rayon 可能把矩陣運算分配到慢速核心 → **慢 ~43%**。

### 2. 使用 Q4 量化（`--quant q4`）

```bash
# Q4 模型 169MB，比 Q8 的 206MB 更小。預測快 16%，品質相當。
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --quant q4
```

指定 `--quant q4` 後會自動從 HuggingFace Hub 下載 Q4 GGUF。

### 3. 跨機器分散部署

```bash
# 機器 1：只跑 talker（獨享 4 顆大核心）
qwen3-tts worker -r talker -b 0.0.0.0:9090

# 機器 2：predictor + vocoder（獨享另外 4 顆大核心）
qwen3-tts worker -r predictor -b 0.0.0.0:9091 --quant q4
qwen3-tts worker -r vocoder -b 0.0.0.0:9092
```

編輯 `qwen3-tts.toml` 指向遠端 worker：
```toml
[workers.talker]
host = "192.168.1.10"   # 機器 1
port = 9090

[workers.predictor]
host = "192.168.1.11"   # 機器 2
port = 9091

[workers.vocoder]
host = "192.168.1.11"   # 機器 2
port = 9092
```

消除核心競爭，RTF 再降 ~10%。

### 4. 編譯時啟用 SDOT（ARM dotprod）

```bash
# 啟用 ARM SDOT 指令，加速量化矩陣乘法
RUSTFLAGS='-C target-feature=+dotprod' cargo build --release
```

Cortex-A76 及更新的核心必備。否則量化推理使用較慢的 vmull+vpaddl 路徑。

### 累積效果

| 優化 | MEDIUM RTF | 加速 |
|------|-----------|------|
| 預設（無優化） | 4.96× | 基準 |
| + 自動大核心綁定 + Q4 + SDOT | 2.87× | **快 42%** |
| + 雙機分散部署 | **2.61×** | **快 47%** |

## 效能

於 RK3588 (4×A76 + 4×A55) 測試，使用預設自動大核心綁定 + `--quant q4`：

### 單機 vs 雙機

| 測試 | 1× RK3588 | 2× RK3588 (LAN) | 提升 |
|------|-----------|------------------|------|
| SHORT (~40 tokens) | 3.30× | **2.99×** | 9% |
| MEDIUM (~175 tokens) | 2.87× | **2.61×** | 9% |
| LONG (200 tokens) | 3.31× | **2.99×** | 10% |

第二台機器分擔 predictor + vocoder，讓第一台 4 顆 A76 專供 talker 使用。
單機部署簡單方便；加一台可降低約 10% RTF。

| 指標 | 1 機 | 2 機 |
|------|------|------|
| Token 生成速率 | ~4.9 tok/s | **~5.5 tok/s** |
| Talker 延遲 | ~35ms/step | ~35ms/step |
| Predictor 延遲 | ~93ms/step | ~93ms/step |
| Vocoder (ONNX FP32) | ~4.5s/batch (無雜音) | ~4.5s/batch |
| Vocoder (RKNN INT8) | ~2.7s/batch (⚠️ 有雜音) | ~2.7s/batch |
| **最佳 MEDIUM RTF** | 2.87× | **2.61×** |
| 外部依賴 | 僅需 `libonnxruntime.so` | 同上 |

> RTF = 生成時間 / 音訊時間。越低越好，RTF < 1 為即時。
>
> **重要：** 在 big.LITTLE SoC（如 RK3588）上，現在預設已啟用大核心綁定。
> 如需手動覆蓋，請使用 `--cores`；否則 predictor 可能從 93ms 退化到 ~190ms。

### 優化歷程

| 優化項目 | Predictor (ms/step) | MEDIUM RTF | 變化 |
|---|---|---|---|
| 基準 (Candle Q8_0, 完整 1.3GB GGUF) | 185 | 4.96× | — |
| + 伺服器端 past_tokens + mem::take() | 185 | 4.79× | −3% |
| + **精簡版 code-predictor GGUF (206MB)** | **108** | **3.12×** | **−37%** |
| + **CPU 親和性（自動大核心綁定）** | **108** | **3.08×** | **−1%** |
| + Q4_0 量化 (`--quant q4`) | 93 | ~2.80× | −3% |
| + **型別化傳輸協議** (ResponseData enum) | 93 | **2.58×** | **−8%** |

關鍵發現：完整 1.3GB GGUF 包含 1075MB 的 talker 權重，但 predictor 從未使用。
精簡至 206MB 的 code-predictor 專用 GGUF 消除了 L2 快取污染 → **預測速度提升 41%**。

> **Q4 vs Q8:** Q4 預測速度提升 ~16% (93ms vs 107ms)，品質相當。
> 使用 `--quant q4` 啟動 predictor worker 以獲得更快速度，`--quant q8` (預設) 較為保守。

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
