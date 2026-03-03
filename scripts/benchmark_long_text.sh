#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_BIN="${QWEN_BIN:-$ROOT_DIR/target/release/qwen3-tts}"
ASR_BIN="${ASR_BIN:-/home/kautism/sensevoice-rs/target/release/examples/basic}"
RUNS="${RUNS:-3}"
LANGUAGE="${LANGUAGE:-chinese}"
CHUNK_MODE="${CHUNK_MODE:-2}"
OUT_DIR="${OUT_DIR:-/tmp/qwen3-bench-$(date +%Y%m%d_%H%M%S)}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <text-file> [runs]" >&2
  exit 1
fi

TEXT_FILE="$1"
if [[ ! -f "$TEXT_FILE" ]]; then
  echo "Text file not found: $TEXT_FILE" >&2
  exit 1
fi

if [[ $# -ge 2 ]]; then
  RUNS="$2"
fi

if [[ ! -x "$QWEN_BIN" ]]; then
  echo "qwen3-tts binary not found: $QWEN_BIN" >&2
  exit 1
fi

if [[ ! -x "$ASR_BIN" ]]; then
  echo "sensevoice ASR binary not found: $ASR_BIN" >&2
  exit 1
fi

if pgrep -f "qwen3-tts serve --port" >/dev/null 2>&1; then
  echo "Found running serve process; stop it before benchmark for isolation." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
TEXT="$(cat "$TEXT_FILE")"
RESULTS_CSV="$OUT_DIR/results.csv"
echo "run,rtf,tokens,audio_s,total_ms,tok_s,asr_content" > "$RESULTS_CSV"

echo "Benchmark output: $OUT_DIR"
echo "Runs=$RUNS language=$LANGUAGE chunk=$CHUNK_MODE"

for ((i=1; i<=RUNS; i++)); do
  WAV="$OUT_DIR/run_${i}.wav"
  WAV16="$OUT_DIR/run_${i}_16k.wav"
  LOG="$OUT_DIR/run_${i}.log"

  echo "[run $i/$RUNS] synthesizing..."
  set +e
  "$QWEN_BIN" speak "$TEXT" --lang "$LANGUAGE" --chunk "$CHUNK_MODE" -o "$WAV" >"$LOG" 2>&1
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "Run $i failed in synthesis. See: $LOG" >&2
    exit 10
  fi

  DONE_LINE="$(grep -E 'Done: .*RTF=' "$LOG" | tail -n1 || true)"
  SAVE_LINE="$(grep -E 'Saved .*tokens' "$LOG" | tail -n1 || true)"
  if [[ -z "$DONE_LINE" || -z "$SAVE_LINE" ]]; then
    echo "Run $i missing parseable metrics. See: $LOG" >&2
    exit 11
  fi

  RTF="$(sed -n 's/.*RTF=\([0-9.]\+\)x.*/\1/p' <<< "$DONE_LINE")"
  AUDIO_S="$(sed -n 's/.*Done: \([0-9.]\+\)s audio in.*/\1/p' <<< "$DONE_LINE")"
  TOKENS="$(sed -n 's/.*audio, \([0-9]\+\) tokens.*/\1/p' <<< "$SAVE_LINE")"
  TOTAL_MS="$(sed -n 's/.*,\s*\([0-9]\+\)ms).*/\1/p' <<< "$SAVE_LINE")"
  if [[ -z "$RTF" || -z "$AUDIO_S" || -z "$TOKENS" || -z "$TOTAL_MS" ]]; then
    echo "Run $i parse failure. See: $LOG" >&2
    exit 12
  fi

  TOK_S="$(python - <<PY
tokens=$TOKENS
total_ms=$TOTAL_MS
print(f"{(tokens/(total_ms/1000.0)):.2f}" if total_ms > 0 else "0.00")
PY
)"

  echo "[run $i/$RUNS] ASR..."
  ffmpeg -y -i "$WAV" -ar 16000 -ac 1 "$WAV16" >/dev/null 2>&1
  set +e
  ASR_RAW="$("$ASR_BIN" "$WAV16" 2>&1)"
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "Run $i failed in ASR. See: $WAV16" >&2
    exit 20
  fi
  ASR_CONTENT="$(tail -n1 <<< "$ASR_RAW" | tr '\n' ' ' | sed 's/"/""/g')"

  printf '%s,%s,%s,%s,%s,%s,"%s"\n' \
    "$i" "$RTF" "$TOKENS" "$AUDIO_S" "$TOTAL_MS" "$TOK_S" "$ASR_CONTENT" >> "$RESULTS_CSV"
done

python - <<PY
import csv
rows=[]
with open("$RESULTS_CSV", newline="", encoding="utf-8") as f:
    r=csv.DictReader(f)
    rows=list(r)
avg_rtf=sum(float(x["rtf"]) for x in rows)/len(rows)
avg_tok=sum(float(x["tok_s"]) for x in rows)/len(rows)
print(f"Completed {len(rows)} runs")
print(f"Average RTF: {avg_rtf:.3f}x")
print(f"Average tok/s: {avg_tok:.3f}")
print(f"CSV: $RESULTS_CSV")
PY
