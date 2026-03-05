#!/usr/bin/env bash
set -euo pipefail

RAW_BASE="https://github.com/darkautism/qwen3-tts/raw/refs/heads/master/deploy/systemd"
GIT_REPO="https://github.com/darkautism/qwen3-tts"
SERVICE="qwen3-tts-frontend.service"
UNIT_DIR="${HOME}/.config/systemd/user"
CFG_DIR="${HOME}/.config/qwen3-tts"
CFG_FILE="${CFG_DIR}/config.toml"
ENV_FILE="${CFG_DIR}/systemd.env"
CARGO_HOME="${HOME}/.cargo"
CARGO_BIN="${CARGO_HOME}/bin"
QWEN3_TTS_BIN="${CARGO_BIN}/qwen3-tts"

export PATH="${CARGO_BIN}:${PATH}"

if [[ ! -x "${QWEN3_TTS_BIN}" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found. Install Rust first: https://rustup.rs" >&2
    exit 1
  fi
  cargo install --git "${GIT_REPO}" --locked --root "${CARGO_HOME}" qwen3-tts-rs
fi

if [[ ! -x "${QWEN3_TTS_BIN}" ]]; then
  echo "qwen3-tts not found at ${QWEN3_TTS_BIN}" >&2
  exit 1
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
systemctl --user enable "${SERVICE}"

if [[ -f "${CFG_FILE}" ]]; then
  systemctl --user restart "${SERVICE}"
else
  echo "Config missing: ${CFG_FILE}"
  echo "Generate config first, then start service:"
  echo "  qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP2>"
  echo "  systemctl --user start ${SERVICE}"
fi

if command -v loginctl >/dev/null 2>&1; then
  if [[ "$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || echo no)" != "yes" ]]; then
    loginctl enable-linger "${USER}" >/dev/null 2>&1 || echo "Run once: sudo loginctl enable-linger ${USER}"
  fi
fi

echo "Installed ${SERVICE}"
