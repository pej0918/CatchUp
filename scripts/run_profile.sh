#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE=llava15; BUDGET=64; GPU=""; PASSES=1; WARMUP=0; DECODE=64; OUT=""
while (($#)); do
 case "$1" in
 --backbone|--budget|--gpu|--n-passes|--n-warmup-passes|--n-decode-tokens|--out)
   if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
   case "$1" in
    --backbone) BACKBONE="$2";; --budget) BUDGET="$2";;
    --gpu) GPU="$2";; --n-passes) PASSES="$2";;
    --n-warmup-passes) WARMUP="$2";; --n-decode-tokens) DECODE="$2";;
    --out) OUT="$2";;
   esac
   shift 2;;
 -h|--help) echo "Usage: bash scripts/run_profile.sh [--backbone llava15|qwen25vl] [--budget 64|128|192] [--gpu ID] [--n-passes N] [--n-warmup-passes N] [--out PATH]"; exit 0;;
 *) echo "Unknown argument: $1" >&2; exit 2;;
 esac
done
case "$BACKBONE:$BUDGET" in
 llava15:64|llava15:128|llava15:192) CONFIG="experiment/llava15/avg${BUDGET}/llava15_qver_avg${BUDGET}";;
 qwen25vl:64|qwen25vl:128|qwen25vl:192) CONFIG="experiment/qwen25vl/avg${BUDGET}/qwen25vl_restore_avg${BUDGET}";;
 *) echo "Unsupported backbone/budget" >&2; exit 2;;
esac
source "$ROOT/scripts/catchup_env.sh"
if [[ -n "$GPU" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
if [[ -z "$OUT" ]]; then OUT="$ROOT/experiments/profile/${BACKBONE}_avg${BUDGET}.json"; fi
mkdir -p "$(dirname "$OUT")"
cd "$ROOT"
python profiler/measure_metrics.py --config-name "$CONFIG" --n-passes "$PASSES" --n-warmup-passes "$WARMUP" --out "$OUT"
