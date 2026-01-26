#!/usr/bin/env bash
set -euo pipefail

# DAC + X-Codec experiment matrix (small/medium/large) + common-ground eval.
#
# Prereqs:
#   export EGMD_CSV=".../e-gmd-v1.0.0.csv"
#   export EGMD_ROOT=".../e-gmd-v1.0.0"
#
# Usage:
#   ENCODE_DEVICE=cuda:2 DEVICE=cuda:2 DECODE_DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh cache
#   ENCODE_DEVICE=cuda:2 DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh train
#   DEVICE=cuda:2 DECODE_DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh eval
#   ENCODE_DEVICE=cuda:2 DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh all
#   ENCODE_DEVICE=cuda:2 DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh screen
#
# Notes:
# - "small" uses d_model=356,n_heads=4 (d_model must be divisible by n_heads).

CMD="${1:-}"
if [[ -z "${CMD}" || "${CMD}" == "-h" || "${CMD}" == "--help" ]]; then
  echo "usage: $0 <cache|train|eval|all|screen>"
  exit 0
fi

if [[ -z "${EGMD_CSV:-}" || -z "${EGMD_ROOT:-}" ]]; then
  echo "error: set EGMD_CSV and EGMD_ROOT in your environment" >&2
  exit 2
fi

ENCODE_DEVICE="${ENCODE_DEVICE:-cuda:0}"
DEVICE="${DEVICE:-cuda:0}"
DECODE_DEVICE="${DECODE_DEVICE:-$DEVICE}"

STEPS="${STEPS:-200000}"
LOG_EVERY="${LOG_EVERY:-300}"
EARLY_STOP_STEPS="${EARLY_STOP_STEPS:-5000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}" # 0 => all validation batches

MAX_ITEMS="${MAX_ITEMS:-256}"
SPLIT="${SPLIT:-test}"
EVAL_SR="${EVAL_SR:-16000}"

# Cache directories (must be produced by `python -m midigroove_poc drumgrid train --precache --precache-only ...`)
CACHE_DAC="${CACHE_DAC:-cache/dac_big_acoustic_pop}"
CACHE_XCODEC="${CACHE_XCODEC:-cache/xcodec_big_acoustic_pop_bw2.0}"
XCODEC_BW="${XCODEC_BW:-2.0}"

# Output checkpoints
CKPT_DIR="${CKPT_DIR:-artifacts/checkpoints}"

COMMON_CACHE_ARGS=(
  --train-csv "$EGMD_CSV"
  --val-csv "$EGMD_CSV"
  --train-split train
  --val-split validation
  --test-split test
  --dataset-root "$EGMD_ROOT"
  --precache --precache-only
  --beat-type-only beat
  --kit-category-only "Acoustic/Pop"
  --stratify-clips
  --encode-device "$ENCODE_DEVICE"
  --num-workers 0
  --seed 0
  --beats-per-chunk 4 --hop-beats 4
)

run_cache() {
  echo "==> building cache: dac -> ${CACHE_DAC}"
  python -m midigroove_poc drumgrid train \
    --cache-dir "$CACHE_DAC" \
    --encoder-model dac \
    "${COMMON_CACHE_ARGS[@]}"

  echo "==> building cache: xcodec(bw=${XCODEC_BW}) -> ${CACHE_XCODEC}"
  python -m midigroove_poc drumgrid train \
    --cache-dir "$CACHE_XCODEC" \
    --encoder-model xcodec \
    --xcodec-bandwidth "$XCODEC_BW" \
    "${COMMON_CACHE_ARGS[@]}"
}

train_one() {
  local codec="$1"
  local cache="$2"
  local size="$3"
  local d_model="$4"
  local n_layers="$5"
  local n_heads="$6"
  local batch_size="$7"

  local save="${CKPT_DIR}/expgrid_${codec}_${size}.pt"

  echo "==> train ${codec} ${size}: d_model=${d_model} n_layers=${n_layers} n_heads=${n_heads} batch=${batch_size}"
  python -m midigroove_poc expressivegrid train \
    --cache "$cache" \
    --device "$DEVICE" \
    --save "$save" \
    --encoder-model "$codec" \
    --steps "$STEPS" \
    --log-every "$LOG_EVERY" \
    --early-stop-steps "$EARLY_STOP_STEPS" \
    --eval-max-batches "$EVAL_MAX_BATCHES" \
    --batch-size "$batch_size" \
    --d-model "$d_model" \
    --n-layers "$n_layers" \
    --n-heads "$n_heads"
}

run_train() {
  mkdir -p "$CKPT_DIR"

  # small / medium / large
  # small uses d_model=360 to satisfy d_model % n_heads == 0 (requested 356 isn't divisible by 6).
  for codec in dac xcodec; do
    cache="$CACHE_DAC"
    [[ "$codec" == "xcodec" ]] && cache="$CACHE_XCODEC"

    train_one "$codec" "$cache" small  356  4  4  32
    train_one "$codec" "$cache" medium 768  6  8  24
    train_one "$codec" "$cache" large  2048 12 16 4
  done
}

run_eval() {
  mkdir -p artifacts/exp
  local systems_file="artifacts/exp/systems_dac_xcodec.txt"
  cat >"$systems_file" <<EOF
# name:ckpt:cache[:encoder_model]
dac_small:${CKPT_DIR}/expgrid_dac_small.pt:${CACHE_DAC}:dac
dac_medium:${CKPT_DIR}/expgrid_dac_medium.pt:${CACHE_DAC}:dac
dac_large:${CKPT_DIR}/expgrid_dac_large.pt:${CACHE_DAC}:dac
xcodec_small:${CKPT_DIR}/expgrid_xcodec_small.pt:${CACHE_XCODEC}:xcodec
xcodec_medium:${CKPT_DIR}/expgrid_xcodec_medium.pt:${CACHE_XCODEC}:xcodec
xcodec_large:${CKPT_DIR}/expgrid_xcodec_large.pt:${CACHE_XCODEC}:xcodec
EOF

  local out_dir="artifacts/eval/dac_xcodec_matrix_$(date +%Y%m%d_%H%M%S)"
  echo "==> eval on common ground (intersection across caches), out_dir=${out_dir}"
  python -m midigroove_poc eval \
    --systems-file "$systems_file" \
    --intersection \
    --split "$SPLIT" \
    --max-items "$MAX_ITEMS" \
    --device "$DEVICE" \
    --decode-device "$DECODE_DEVICE" \
    --eval-sr "$EVAL_SR" \
    --audio-metrics \
    --add-oracle \
    --out-dir "$out_dir"
}

run_all() {
  run_cache
  run_train
  if [[ "${DO_EVAL:-0}" == "1" ]]; then
    run_eval
  fi
}

run_screen() {
  if ! command -v screen >/dev/null 2>&1; then
    echo "error: GNU screen is not installed (install it or run '$0 all' directly)" >&2
    exit 2
  fi
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local session="${SCREEN_SESSION:-midigroove_exp_${ts}}"
  local log_dir="${SCREEN_LOG_DIR:-artifacts/exp/logs}"
  mkdir -p "$log_dir"
  local log_file="${log_dir}/${session}.log"

  echo "==> starting screen session: ${session}"
  echo "==> log: ${log_file}"
  screen -S "$session" -Logfile "$log_file" -Log -dm bash -lc "set -euo pipefail; cd \"$(pwd)\"; bash scripts/experiments_dac_xcodec.sh all"
  echo "attach with: screen -r ${session}"
}

case "$CMD" in
  cache) run_cache ;;
  train) run_train ;;
  eval)  run_eval ;;
  all)   run_all ;;
  screen) run_screen ;;
  *)
    echo "unknown command: ${CMD} (expected cache|train|eval|all|screen)" >&2
    exit 2
    ;;
esac
