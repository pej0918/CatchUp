#!/usr/bin/env bash
# Evaluate prefill TFLOPs / KV cache / peak memory or end-to-end latency.
# Example: bash scripts/run_efficiency.sh --mode runtime --budget 64 --gpu 0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mode=runtime; budget=64; gpu=0; compare=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="$2"; shift 2;;
    --budget) budget="$2"; shift 2;;
    --gpu) gpu="$2"; shift 2;;
    --compare) compare="$2"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
case "$mode" in profile|runtime);; *) echo "--mode: profile or runtime" >&2; exit 2;; esac
case "$budget" in 64|128|192|all);; *) echo "--budget: 64|128|192|all" >&2; exit 2;; esac
case "$compare" in all|catchup|dense|without-ltc);; *) echo "--compare: all|catchup|dense|without-ltc" >&2; exit 2;; esac
source "$ROOT/scripts/catchup_env.sh"
export CUDA_VISIBLE_DEVICES="$gpu"
CONDA_ENV="${CONDA_ENV:-reroute}"
budgets=("$budget")
if [[ "$budget" == all ]]; then budgets=(64 128 192); fi
mkdir -p "experiments/${mode}"
for b in "${budgets[@]}"; do
  for variant in dense without-ltc catchup; do
    if [[ "$compare" != all && "$compare" != "$variant" ]]; then continue; fi
    cfg="experiment/llava15/avg${b}/llava15_qver_avg${b}"
    correction=qver
    if [[ "$variant" == dense ]]; then
      cfg="experiment/baseline/llava15"
      correction=none
    elif [[ "$variant" == without-ltc ]]; then
      correction=none
    fi
    out="experiments/${mode}/${variant}_avg${b}.json"
    if [[ "$mode" == runtime ]]; then
      REROUTE_CORRECTION_MODE="$correction" conda run --no-capture-output -n "$CONDA_ENV" \
        python profiler/bench_runtime.py --config-name "$cfg" \
        --n-passes 5 --n-warmup-passes 2 --n-decode-tokens 64 --out "$out"
    else
      REROUTE_CORRECTION_MODE="$correction" conda run --no-capture-output -n "$CONDA_ENV" \
        python profiler/measure_metrics.py --config-name "$cfg" \
        --n-passes 1 --n-warmup-passes 0 --out "$out"
    fi
  done
done
