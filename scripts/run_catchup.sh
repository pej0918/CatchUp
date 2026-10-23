#!/usr/bin/env bash
# Run one CatchUp/LLaVA benchmark. Requires lmms-eval and its grounding patches.
# Example: bash scripts/run_catchup.sh --budget 64 --task pope --gpu 0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
budget=64
task=pope
gpu=0
limit=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) budget="$2"; shift 2;;
    --task) task="$2"; shift 2;;
    --gpu) gpu="$2"; shift 2;;
    --limit) limit="$2"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
case "$budget" in 64|128|192);; *) echo "Budget must be 64, 128 or 192" >&2; exit 2;; esac
source "$ROOT/scripts/catchup_env.sh"
cfg="llava15/avg${budget}/llava15_qver_avg${budget}"
args=("$cfg" --tasks "$task" --gpu "$gpu")
if [[ -n "$limit" ]]; then args+=(--limit "$limit"); fi
bash "$ROOT/scripts/run_setting.sh" "${args[@]}"
