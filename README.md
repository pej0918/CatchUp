<div align="center">

# 🪃 CatchUp

<h3>Depth-Aware Re-Admission of Visual Tokens<br>for Efficient Large Vision-Language Models</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2610.23324-b31b1b.svg)](https://arxiv.org/abs/2610.23324)
[![WACV](https://img.shields.io/badge/WACV-2027-1f6feb.svg)](https://wacv.thecvf.com)

<!-- Add when a real project page and a LICENSE file are available:
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f.svg)](YOUR_PROJECT_PAGE_LINK)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
-->

</div>

> **CatchUp** is a training-free framework for efficient LVLM inference that combines **reversible visual-token routing** with **Local Transition Correction (LTC)** to improve the representation readiness of re-admitted tokens.

## 🗂️ Layout

```text
models/                    Q-VER routing and LTC
configs/                   Model and benchmark configurations
scripts/                   Evaluation and efficiency scripts
profiler/                  Runtime, FLOPs, and memory profiling
lmms_eval_patches/         RefCOCO evaluation patches
bench_data/                Efficiency benchmark samples
```

## 🛠️ Installation

Run from the CatchUp repository root. The PyTorch command below is an example for **CUDA 12.8**; choose a compatible build from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for other CUDA environments.

```bash
conda create -n reroute python=3.10 -y
conda activate reroute

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e ./lmms-eval
pip install -r requirements.txt
bash lmms_eval_patches/apply.sh

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
```

## 🚀 Quick Start

**Evaluation**

```bash
# Smoke test (2 samples per task)
bash scripts/run_catchup.sh --budget 64 --task 'pope,refcoco+_val' --gpu 0 --limit 2

# POPE and RefCOCO+ Val (K=192)
bash scripts/run_catchup.sh --budget 192 --task 'pope,refcoco+_val' --gpu 0

# All RefCOCO / RefCOCO+ / RefCOCOg splits (K=64)
bash scripts/run_catchup.sh --budget 64 --task grounding --gpu 0
```

**Efficiency**

```bash
bash scripts/run_efficiency.sh --mode runtime --budget 64 --gpu 0 --compare catchup
bash scripts/run_efficiency.sh --mode profile --budget 64 --gpu 0 --compare catchup
```

Available budget profiles: `64`, `128`, and `192`. TFLOPs profiling additionally requires DeepSpeed (see `requirements.txt`).

## 📝 Citation

```bibtex
@inproceedings{park2027catchup,
  title={CatchUp: Depth-Aware Re-Admission of Visual Tokens for Efficient Large Vision-Language Models},
  author={Eunju Park and Uiwon Hwang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```
