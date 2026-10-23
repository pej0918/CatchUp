# Efficiency benchmark cohort

This folder contains **three fixed COCO val2014 images**, provided through the `lmms-lab/POPE` test split (see the upstream dataset terms, CC-BY 4.0). The same cohort is used for prefill TFLOPs/KV memory and latency comparisons.

| Sample | Image ID | Dimensions |
| --- | --- | --- |
| 0 | `COCO_val2014_000000310196` | 640×427 |
| 1 | `COCO_val2014_000000210789` | 369×520 |
| 2 | `COCO_val2014_000000429109` | 640×427 |

Prompt: `Describe this image in detail.`

Each entry in `manifest.json` contains `sample_id`, `image_id`, `category`, `question`, `image`, and `image_WxH`.

## Supported commands

From the repository root:

```bash
# Runtime / prefill / decode, with five timed passes + two warmup passes
bash scripts/run_efficiency.sh --mode runtime --budget 64 --gpu 0 --compare catchup

# DeepSpeed prefill profiler, one measurement per image
bash scripts/run_efficiency.sh --mode profile --budget 64 --gpu 0 --compare catchup

# Dense, Q-VER without LTC, and CatchUp profiles for the same budget
bash scripts/run_efficiency.sh --mode profile --budget 192 --gpu 0 --compare all
```

Results are written under `experiments/runtime/` and `experiments/profile/`.

For advanced runs, the corresponding entrypoints are `profiler/bench_runtime.py` and `profiler/measure_metrics.py`, each requiring `--config-name` and `--out`. The runtime script also supports `--n-passes`, `--n-warmup-passes`, and `--n-decode-tokens`; the prefill profiler does not decode. See `--help` for details.

## Replacing the cohort

Replace the images and `manifest.json` with your own test samples. The manifest must remain a list of objects with the same keys and the expected image dimensions. Report any replacement cohort and generation settings explicitly when comparing efficiency numbers.
