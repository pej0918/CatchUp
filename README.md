# CatchUp: Depth-Aware Re-Admission of Visual Tokens for Efficient Large Vision-Language Models

**CatchUp** is a training-free framework for efficient large vision-language models that combines reversible visual-token routing with **Local Transition Correction (LTC)** to correct stale token representations upon re-admission.

## Layout

~~~text
models/               reversible routing, deferred-token dispatch, LTC, baseline routers
configs/experiment/   LLaVA / Qwen, routing budgets 64/128/192
configs/eval/         POPE, grounding, and MLLM evaluation definitions
configs/model/        model identifiers and model-specific settings
scripts/run_eval.py   Hydra + lmms-eval entry point
scripts/run_catchup.sh             single benchmark entry point
scripts/run_paper_benchmarks.sh    six benchmark groups / three budgets
scripts/catchup_env.sh             canonical release LTC settings
scripts/run_profile.sh             prefill TFLOPs / memory
scripts/run_runtime_bench.sh       runtime / latency
profiler/             efficiency measurement implementations
bench_data/           fixed profiling cohort (not accuracy benchmarks)
lmms_eval_patches/    RefCOCO evaluation compatibility patches
~~~

## Installation

The original code targets Python 3.10 and the dependencies in `requirements.txt`. Use CUDA-compatible PyTorch / torchvision wheels for your machine before installing the remaining dependencies.

~~~bash
conda create -n catchup python=3.10 -y
conda activate catchup
# Install torch and torchvision appropriate for your CUDA installation.
pip install -r requirements.txt
git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e ./lmms-eval
bash lmms_eval_patches/apply.sh
~~~

A CUDA GPU is required to evaluate the 7B backbones. MM-Vet grading may require an `OPENAI_API_KEY`, depending on the evaluator.

## Quick start

~~~bash
# CatchUp on LLaVA-1.5-7B, budget 64, POPE.
bash scripts/run_catchup.sh --backbone llava15 --budget 64 --eval pope --gpu 0

# All RefCOCO/+/g grounding splits and POPE.
bash scripts/run_catchup.sh --backbone llava15 --budget 128 --eval paper_main --gpu 0

# Quick smoke test.
bash scripts/run_catchup.sh --backbone llava15 --budget 64 --eval pope --limit 10

# Second backbone.
bash scripts/run_catchup.sh --backbone qwen25vl --budget 64 --eval pope --gpu 0

# Sequentially run paper benchmark groups at all three budgets.
bash scripts/run_paper_benchmarks.sh --backbone llava15 --gpu 0
~~~

The scripts write results to `experiments/paper/` (git-ignored). For locally downloaded model weights, pass `--model-path /path/to/checkpoint` to `run_catchup.sh`. Individual task configs can be selected by name under `configs/eval/` without the `.yaml` suffix.
