#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE=llava15; GPU=""; MODEL_PATH=""; LIMIT=""
while (($#)); do
 case "$1" in
 --backbone|--gpu|--model-path|--limit)
   if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
   case "$1" in
    --backbone) BACKBONE="$2";; --gpu) GPU="$2";;
    --model-path) MODEL_PATH="$2";; --limit) LIMIT="$2";;
   esac
   shift 2;;
 -h|--help) echo "Usage: bash scripts/run_paper_benchmarks.sh [--backbone llava15|qwen25vl] [--gpu ID] [--model-path PATH] [--limit N]"; exit 0;;
 *) echo "Unknown argument: $1" >&2; exit 2;;
 esac
done
TASKS=(paper_main gqa mmbench mmvet vizwiz_vqa_val vqav2_val)
for BUDGET in 64 128 192; do
 for TASK in "${TASKS[@]}"; do
  ARGS=(--backbone "$BACKBONE" --budget "$BUDGET" --eval "$TASK")
  if [[ -n "$GPU" ]]; then ARGS+=(--gpu "$GPU"); fi
  if [[ -n "$MODEL_PATH" ]]; then ARGS+=(--model-path "$MODEL_PATH"); fi
  if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi
  bash "$ROOT/scripts/run_catchup.sh" "${ARGS[@]}"
 done
done
