#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE=llava15; BUDGET=64; EVAL=pope; GPU=""; LIMIT=""; MODEL_PATH=""; OUTPUT_DIR=""
while (($#)); do
  case "$1" in
    --backbone|--budget|--eval|--gpu|--limit|--model-path|--output-dir)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --backbone) BACKBONE="$2";; --budget) BUDGET="$2";;
        --eval) EVAL="$2";; --gpu) GPU="$2";;
        --limit) LIMIT="$2";; --model-path) MODEL_PATH="$2";;
        --output-dir) OUTPUT_DIR="$2";;
      esac
      shift 2;;
    -h|--help)
      echo "Usage: bash scripts/run_catchup.sh [--backbone llava15|qwen25vl] [--budget 64|128|192] [--eval TASK] [--gpu ID] [--limit N] [--model-path ID_OR_PATH] [--output-dir DIR]"
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
case "$BACKBONE:$BUDGET" in
  llava15:64|llava15:128|llava15:192) CONFIG="llava15/avg${BUDGET}/llava15_qver_avg${BUDGET}";;
  qwen25vl:64|qwen25vl:128|qwen25vl:192) CONFIG="qwen25vl/avg${BUDGET}/qwen25vl_restore_avg${BUDGET}";;
  *) echo "Unsupported backbone/budget" >&2; exit 2;;
esac
if [[ ! "$EVAL" =~ ^[a-zA-Z0-9_+.-]+$ || ! -f "$ROOT/configs/eval/${EVAL}.yaml" ]]; then echo "Invalid eval: $EVAL" >&2; exit 2; fi
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid limit" >&2; exit 2; fi
source "$ROOT/scripts/catchup_env.sh"
if [[ -n "$GPU" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
if [[ -z "$OUTPUT_DIR" ]]; then OUTPUT_DIR="$ROOT/experiments/paper/$BACKBONE/avg$BUDGET/$EVAL"; fi
mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CMD=(python "$ROOT/scripts/run_eval.py" --config-dir "$ROOT/configs" --config-name "experiment/$CONFIG" "eval=$EVAL" "hydra.run.dir=$OUTPUT_DIR")
if [[ -n "$LIMIT" ]]; then CMD+=("+eval.limit=$LIMIT"); fi
if [[ -n "$MODEL_PATH" ]]; then CMD+=("model.pretrained=$MODEL_PATH"); fi
cd "$ROOT"
HYDRA_FULL_ERROR=1 "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
