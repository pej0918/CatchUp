# Efficiency profiling cohort

This folder includes three COCO val2014 images sourced through the `lmms-lab/POPE` test split. See upstream dataset terms for source licensing; identifiers and image sizes are recorded in `manifest.json`. The shared prompt is "Describe this image in detail."

These samples are used only by the TFLOPs/memory and latency profilers, **not** for the paper's accuracy benchmarks.

~~~bash
bash scripts/run_profile.sh --backbone llava15 --budget 64 --gpu 0
bash scripts/run_runtime_bench.sh --backbone llava15 --budget 64 --gpu 0
~~~

To replace the cohort, update the manifest and images while keeping the keys: `sample_id`, `image_id`, `category`, `question`, `image`, and `image_WxH`.
