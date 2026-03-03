#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DST="${HOME}/.config/systemd/user"
CFG_DIR="${HOME}/.config/qwen3-tts"
ENV_FILE="${CFG_DIR}/systemd.env"

mkdir -p "${UNIT_DST}" "${CFG_DIR}"
cp "${SCRIPT_DIR}"/qwen3-tts-*.service "${UNIT_DST}/"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
# Shared overrides for qwen3-tts user services
# Binary path:
QWEN3_TTS_BIN=/home/YOUR_USER/.local/bin/qwen3-tts

# Role endpoints:
QWEN3_TTS_TALKER_BIND=0.0.0.0:9090
QWEN3_TTS_PREDICTOR_BIND=0.0.0.0:9091
QWEN3_TTS_VOCODER_BIND=0.0.0.0:9092
QWEN3_TTS_SERVE_PORT=8080

# Quality-safe best-RTF defaults:
QWEN3_TTS_PREDICTOR_QUANT=q4
QWEN3_TTS_REPO=kautism/qwen3-tts-rk3588
EOF
  sed -i "s#/home/YOUR_USER#${HOME}#g" "${ENV_FILE}"
fi

systemctl --user import-environment HOME XDG_CONFIG_HOME PATH || true
systemctl --user daemon-reload

cat <<EOF
Installed user units to: ${UNIT_DST}
Environment overrides:   ${ENV_FILE}

Before starting, ensure config exists:
  ${CFG_DIR}/config.toml
Generate one if needed:
  qwen3-tts init --talker-ip <IP1> --predictor-ip <IP2> --vocoder-ip <IP2>

Enable/start example (main node: talker + frontend):
  systemctl --user enable --now qwen3-tts-talker.service qwen3-tts-frontend.service

Enable/start example (compute node: predictor + vocoder):
  systemctl --user enable --now qwen3-tts-predictor.service qwen3-tts-vocoder.service

Check logs:
  journalctl --user -u qwen3-tts-frontend.service -f
EOF
