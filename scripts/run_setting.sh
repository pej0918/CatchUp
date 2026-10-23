#!/usr/bin/env bash
# Low-level CatchUp benchmark runner (typically called via run_catchup.sh).
# Usage: bash scripts/run_setting.sh <config_path> [--tasks <selector>] [--gpu N] [--limit N]
# Config: llava15/avg{64,128,192}/llava15_qver_avg{64,128,192}
#         baseline/llava15
# Selectors: pope, refcoco (three RefCOCO splits), grounding (eight splits),
#            paper_main, or comma-separated individual task names.
# Results and per-sample outputs are stored in experiments/logs/<config_path>/.
#
set -euo pipefail

CONFIG="${1:?Usage: $0 <config_path> [--tasks <sel>] [--limit <N>] [--gpu <id>]}"
shift

TASKS_ARG="pope"
LIMIT_ARG=""
GPU_ARG=""
LOG_DIR_BASE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)   TASKS_ARG="$2"; shift 2 ;;
        --limit)   LIMIT_ARG="+eval.limit=$2"; shift 2 ;;
        --gpu)     GPU_ARG="$2"; shift 2 ;;
        --log-dir) LOG_DIR_BASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$GPU_ARG" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ARG"
fi

# --- Task selectors ---
GROUNDING=("refcoco_val" "refcoco_testA" "refcoco_testB"
           "refcoco+_val" "refcoco+_testA" "refcoco+_testB"
           "refcocog_val" "refcocog_test")
REFCOCO=("refcoco_val" "refcoco_testA" "refcoco_testB")
VQA=("gqa" "mmbench" "mme")
# Fast iteration set: 1 general VQA + 1 multi-skill VQA + 2 grounding splits.
ABLATION=("gqa" "mmbench" "refcoco_testA" "refcoco_testB")

case "$TASKS_ARG" in
    pope)       TASKS=("pope") ;;
    refcoco)    TASKS=("${REFCOCO[@]}") ;;
    grounding)  TASKS=("${GROUNDING[@]}") ;;
    vqa)        TASKS=("${VQA[@]}") ;;
    ablation)   TASKS=("${ABLATION[@]}") ;;
    paper_main) TASKS=("paper_main") ;;
    *)          IFS=',' read -ra TASKS <<< "$TASKS_ARG" ;;
esac

CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-reroute}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve log-dir base
if [[ -z "$LOG_DIR_BASE" ]]; then
    LOG_DIR_BASE="${PROJECT_ROOT}/experiments/logs"
elif [[ "$LOG_DIR_BASE" != /* ]]; then
    LOG_DIR_BASE="${PROJECT_ROOT}/${LOG_DIR_BASE}"
fi
LOG_DIR="${LOG_DIR_BASE}/${CONFIG}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Config:    ${CONFIG}"
echo "Tasks:     ${TASKS[*]}"
echo "Log dir:   ${LOG_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "GPU(s):    ${CUDA_VISIBLE_DEVICES:-<inherit/all>}"
[[ -n "$LIMIT_ARG" ]] && echo "Limit:     ${LIMIT_ARG}"
echo "================================================================"

cd "${PROJECT_ROOT}"

ANY_FAILED=0
for TASK in "${TASKS[@]}"; do
    LOG_FILE="${LOG_DIR}/${TASK}.log"
    HYDRA_DIR="${LOG_DIR}/${TASK}"
    echo
    echo "===== $(date '+%F %T'): START ${CONFIG} x ${TASK} ====="

    # lmms-eval may catch model-generation exceptions and exit with status 0.
    # Check both the subprocess exit status and its logged ERROR records.
    set +e
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python scripts/run_eval.py \
        --config-dir "${PROJECT_ROOT}/configs" \
        --config-name "experiment/${CONFIG}" \
        "eval=${TASK}" \
        ${LIMIT_ARG} \
        "hydra.run.dir=${HYDRA_DIR}" \
        2>&1 | tee "${LOG_FILE}"
    STATUS=${PIPESTATUS[0]}
    set -e

    if [[ "$STATUS" -ne 0 ]] || grep -Eq '[|][[:space:]]*ERROR[[:space:]]*[|]|\]\[ERROR\]|Traceback \(most recent call last\)' "${LOG_FILE}"; then
        echo "===== $(date '+%F %T'): FAIL  ${CONFIG} x ${TASK} (exit=${STATUS}; check error log) ====="
        ANY_FAILED=1
    else
        echo "===== $(date '+%F %T'): DONE  ${CONFIG} x ${TASK} ====="
    fi
done

echo
echo "All tasks finished for ${CONFIG}. Logs: ${LOG_DIR}"
exit "$ANY_FAILED"
