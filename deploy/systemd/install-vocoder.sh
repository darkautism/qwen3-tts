#!/usr/bin/env bash
set -euo pipefail

RAW_BASE="https://github.com/darkautism/qwen3-tts/raw/refs/heads/master/deploy/systemd"
GIT_REPO="https://github.com/darkautism/qwen3-tts"
SERVICE="qwen3-tts-vocoder.service"
UNIT_DIR="${HOME}/.config/systemd/user"
CFG_DIR="${HOME}/.config/qwen3-tts"
ENV_FILE="${CFG_DIR}/systemd.env"

if ! command -v qwen3-tts >/dev/null 2>&1; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found. Install Rust first: https://rustup.rs" >&2
    exit 1
  fi
  cargo install --git "${GIT_REPO}" --locked qwen3-tts-rs
fi

mkdir -p "${UNIT_DIR}" "${CFG_DIR}"
curl -fsSL "${RAW_BASE}/${SERVICE}" -o "${UNIT_DIR}/${SERVICE}"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
QWEN3_TTS_REPO=kautism/qwen3-tts-rk3588
QWEN3_TTS_TALKER_BIND=0.0.0.0:9090
QWEN3_TTS_PREDICTOR_BIND=0.0.0.0:9091
QWEN3_TTS_PREDICTOR_QUANT=q4
QWEN3_TTS_VOCODER_BIND=0.0.0.0:9092
QWEN3_TTS_SERVE_PORT=8080
EOF
fi

systemctl --user import-environment HOME XDG_CONFIG_HOME PATH || true
systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE}"

if command -v loginctl >/dev/null 2>&1; then
  if [[ "$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || echo no)" != "yes" ]]; then
    loginctl enable-linger "${USER}" >/dev/null 2>&1 || echo "Run once: sudo loginctl enable-linger ${USER}"
  fi
fi

echo "Installed and started ${SERVICE} (ONNX FP32, quality-safe)"
