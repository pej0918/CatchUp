#!/usr/bin/env python3
"""Prefill-only TFLOPs + KV-cache + LTC snapshot-memory profiler.

Reads the bundled cohort at ``bench_data/`` and runs one prefill forward
(no decode) per sample x n_passes. It reports:

  * DeepSpeed prefill TFLOPs (when DeepSpeed profiler is available)
  * post-prefill KV-cache tokens / bytes
  * peak LTC historical-snapshot cache bytes / count
  * CUDA peak allocated memory and incremental peak above the prefill baseline

The LTC cache statistics are published by ``models/patching.py`` and are reset
for every new visual prefill. This lets FIFO-8, FIFO-32, and live-reference
cache policies be compared without changing the routing or LTC computation.

Usage:
  python profiler/measure_metrics.py \
      --config-name experiment/llava15/avg64/llava15_qver_avg64 \
      --n-passes 1 --n-warmup-passes 0 \
      --out experiments/snapshot_policy/profile/fifo8.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Allow imports from release root (../models)
_RELEASE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RELEASE_ROOT))


def _build_model_and_apply_routing(cfg_path: Path):
    """Load HF model + apply routing patch; expose RoutingContext for profiling."""
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    from models.router import FastVRouter, PDropRouter, QVERouter, SparseVLMRouter
    from models.dispatcher import TokenDispatcher
    from models.patching import patch_model_for_routing

    cfg = OmegaConf.load(cfg_path)

    model_name = "llava15_7b_hf"
    for d in cfg.get("defaults", []):
        if hasattr(d, "keys") and "/model" in d:
            model_name = str(d["/model"])
            break

    model_cfg = OmegaConf.load(
        _RELEASE_ROOT / "configs" / "model" / f"{model_name}.yaml"
    )
    pretrained = model_cfg.pretrained
    lmms_type = model_cfg.get("lmms_model_type", "llava_hf")

    if lmms_type == "llava_hf":
        from transformers import LlavaForConditionalGeneration as ModelCls
        model_family = "llava"
    elif lmms_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        model_family = "qwen25vl"
    elif lmms_type in ("qwen3_vl", "qwen3vl"):
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
        model_family = "qwen3vl"
    else:
        raise ValueError(f"Unknown lmms_model_type: {lmms_type}")

    dtype = torch.float16 if model_family == "llava" else torch.bfloat16
    print(
        f"[PROFILER] loading {pretrained} "
        f"(dtype={dtype}, family={model_family}) ...",
        flush=True,
    )

    load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    if model_family == "llava":
        model = ModelCls.from_pretrained(pretrained, **load_kwargs).to("cuda").eval()
    else:
        load_kwargs["device_map"] = "auto"
        model = ModelCls.from_pretrained(pretrained, **load_kwargs).eval()

    proc_kwargs = {}
    if model_cfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = int(model_cfg.max_pixels)
    processor = AutoProcessor.from_pretrained(pretrained, **proc_kwargs)

    # Keep this handle so the profiler can read LTC cache statistics after
    # each prefill without changing the normal model API.
    routing_ctx = None
    r_cfg = cfg.routing

    if r_cfg.method == "none":
        print("[PROFILER] no routing — vanilla baseline", flush=True)

    elif r_cfg.method == "fastv":
        router = FastVRouter(
            scoring_layer=r_cfg.scoring_layer,
            keep_ratio=r_cfg.keep_ratio,
        )
        routing_ctx = patch_model_for_routing(
            model,
            router,
            TokenDispatcher(),
            action=r_cfg.action,
            model_family=model_family,
        )
        print(
            f"[PROFILER] patched with FastV K={r_cfg.scoring_layer} "
            f"keep={r_cfg.keep_ratio} action={r_cfg.action}",
            flush=True,
        )

    elif r_cfg.method == "pdrop":
        router = PDropRouter(
            drop_layers=list(r_cfg.drop_layers),
            keep_ratios=list(r_cfg.keep_ratios),
            monotonic=r_cfg.get("monotonic", False),
        )
        routing_ctx = patch_model_for_routing(
            model,
            router,
            TokenDispatcher(),
            action=r_cfg.action,
            model_family=model_family,
        )
        print(
            f"[PROFILER] patched with PDrop drops={r_cfg.drop_layers} "
            f"keeps={r_cfg.keep_ratios} action={r_cfg.action}",
            flush=True,
        )

    elif r_cfg.method in ("qver", "query_dynamic"):
        router = QVERouter(
            drop_layers=list(r_cfg.drop_layers),
            keep_ratios=list(r_cfg.keep_ratios),
            monotonic=r_cfg.get("monotonic", False),
        )
        routing_ctx = patch_model_for_routing(
            model,
            router,
            TokenDispatcher(),
            action=r_cfg.action,
            model_family=model_family,
        )
        print(
            f"[PROFILER] patched with Q-VER "
            f"drops={list(r_cfg.drop_layers)} "
            f"keeps={list(r_cfg.keep_ratios)} "
            f"monotonic={r_cfg.get('monotonic', False)} "
            f"action={r_cfg.action}",
            flush=True,
        )

    elif r_cfg.method == "sparsevlm":
        router = SparseVLMRouter(
            retain_tokens=int(r_cfg.get("retain_tokens", 192)),
            use_version=str(r_cfg.get("use_version", "1_0")),
            drop_layers=list(r_cfg.get("drop_layers", [2, 6, 15])),
        )
        routing_ctx = patch_model_for_routing(
            model,
            router,
            TokenDispatcher(),
            action=r_cfg.action,
            model_family=model_family,
        )
        print(
            f"[PROFILER] patched with SparseVLM "
            f"retain_tokens={r_cfg.get('retain_tokens', 192)} "
            f"use_version={r_cfg.get('use_version', '1_0')} "
            f"drops={list(r_cfg.get('drop_layers', [2, 6, 15]))} "
            f"action={r_cfg.action}",
            flush=True,
        )

    elif r_cfg.method == "nuwa":
        from models.nuwa_hf import NuwaRouter

        router = NuwaRouter(
            matain_token=int(r_cfg.get("matain_token", 192)),
            distance=int(r_cfg.get("distance", 280)),
            penalty_threshold_percentile=float(
                r_cfg.get("penalty_threshold_percentile", 0.55)
            ),
            stage1_tokens=int(r_cfg.get("stage1_tokens", 336)),
            stage2_layer=int(r_cfg.get("stage2_layer", 16)),
            stage2_tokens=int(r_cfg.get("stage2_tokens", 48)),
        )
        routing_ctx = patch_model_for_routing(
            model,
            router,
            TokenDispatcher(),
            action=r_cfg.action,
            model_family=model_family,
        )
        print(
            f"[PROFILER] patched with Nuwa "
            f"matain_token={r_cfg.get('matain_token', 192)} "
            f"stage1={r_cfg.get('stage1_tokens', 336)} "
            f"stage2_layer={r_cfg.get('stage2_layer', 16)} "
            f"stage2={r_cfg.get('stage2_tokens', 48)} "
            f"distance={r_cfg.get('distance', 280)} "
            f"action={r_cfg.action}",
            flush=True,
        )

    else:
        raise ValueError(f"Unknown routing.method: {r_cfg.method}")

    model._routing_ctx = routing_ctx
    return model, processor


BENCH_BUNDLED = _RELEASE_ROOT / "bench_data"


def _load_bundled_cohort() -> tuple[Path, list[dict]]:
    manifest_p = BENCH_BUNDLED / "manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(
            f"Bundled bench cohort not found at {BENCH_BUNDLED}. "
            f"Re-extract the release tarball."
        )
    manifest = json.loads(manifest_p.read_text())
    for s in manifest:
        if not (BENCH_BUNDLED / s["image"]).exists():
            raise FileNotFoundError(
                f"Manifest references missing image: {BENCH_BUNDLED / s['image']}"
            )
    return BENCH_BUNDLED, manifest


def _build_inputs_for_sample(processor, model, sample: dict, bench_dir: Path):
    from PIL import Image

    img = Image.open(bench_dir / sample["image"]).convert("RGB")
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["question"]},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(
        model.device,
        dtype=torch.float16 if model.dtype == torch.float16 else torch.bfloat16,
    )
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()
    return inputs


def _walk_kv(out):
    """Sum K+V tokens and bytes across layers; return K seq len per layer."""
    n_tokens, n_bytes, per_layer = 0, 0, []
    kv = out.past_key_values
    if hasattr(kv, "key_cache") and hasattr(kv, "value_cache"):
        iters = zip(kv.key_cache, kv.value_cache)
    else:
        iters = ((lk[0], lk[1]) for lk in kv)

    for k, v in iters:
        n_tokens += k.shape[-2] + v.shape[-2]
        n_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        per_layer.append(int(k.shape[-2]))
    return n_tokens, n_bytes, per_layer


def _snapshot_stats(model) -> dict:
    """Read peak LTC historical-cache statistics for the current prefill."""
    ctx = getattr(model, "_routing_ctx", None)
    if ctx is None:
        return {
            "policy": "none",
            "keep": 0,
            "peak_count": 0,
            "peak_bytes": 0,
            "current_count": 0,
            "current_bytes": 0,
        }

    return {
        "policy": str(getattr(ctx, "correction_snapshot_policy", "fifo")),
        "keep": int(getattr(ctx, "correction_snapshot_keep", 8)),
        "peak_count": int(getattr(ctx, "_correction_snapshot_peak_count", 0)),
        "peak_bytes": int(getattr(ctx, "_correction_snapshot_peak_bytes", 0)),
        "current_count": int(
            getattr(ctx, "_correction_snapshot_current_count", 0)
        ),
        "current_bytes": int(
            getattr(ctx, "_correction_snapshot_current_bytes", 0)
        ),
    }


def _median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[len(ys) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-name",
        required=True,
        help="release experiment config path (e.g. experiment/llava15/avg64/llava15_qver_avg64)",
    )
    ap.add_argument(
        "--n-passes",
        type=int,
        default=3,
        help="measured prefill passes through the bundled cohort",
    )
    ap.add_argument(
        "--n-warmup-passes",
        type=int,
        default=1,
        help="untimed warmup passes",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = _RELEASE_ROOT / "configs" / f"{args.config_name}.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 1

    bench_dir, manifest = _load_bundled_cohort()
    print(
        f"[PROFILER] cohort source: {bench_dir} ({len(manifest)} samples)",
        flush=True,
    )

    model, processor = _build_model_and_apply_routing(cfg_path)
    per_sample_inputs = [
        {
            "sample": s,
            "inputs": _build_inputs_for_sample(processor, model, s, bench_dir),
        }
        for s in manifest
    ]
    for entry in per_sample_inputs:
        entry["prefill_tokens"] = int(entry["inputs"]["input_ids"].shape[1])

    try:
        from deepspeed.profiling.flops_profiler import FlopsProfiler

        prof = FlopsProfiler(model)
        deepspeed_available = True
    except Exception as e:
        prof, deepspeed_available = None, False
        print(
            f"[WARN] deepspeed unavailable ({type(e).__name__}: {e}) — "
            "TFLOPs will be null.",
            flush=True,
        )

    # Warmup: prefill only.
    print(
        f"[PROFILER] warmup {args.n_warmup_passes} pass x "
        f"{len(per_sample_inputs)} samples ...",
        flush=True,
    )
    for _ in range(args.n_warmup_passes):
        for entry in per_sample_inputs:
            with torch.inference_mode():
                warm_out = model(**entry["inputs"], use_cache=True)
            del warm_out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    per_sample_records = [
        {
            "sample_id": entry["sample"]["sample_id"],
            "image_id": entry["sample"].get("image_id"),
            "category": entry["sample"].get("category"),
            "image_WxH": entry["sample"]["image_WxH"],
            "prefill_tokens": entry["prefill_tokens"],
            "tflops": [],
            "kv_cache_tokens": [],
            "kv_cache_bytes": [],
            "ltc_snapshot_peak_count": [],
            "ltc_snapshot_peak_bytes": [],
            "ltc_snapshot_current_count": [],
            "ltc_snapshot_current_bytes": [],
            "cuda_allocated_before_bytes": [],
            "cuda_peak_allocated_bytes": [],
            "cuda_peak_delta_bytes": [],
            "per_layer_kv_seq_len": None,
        }
        for entry in per_sample_inputs
    ]

    print(
        f"[PROFILER] measured {args.n_passes} prefill passes x "
        f"{len(per_sample_inputs)} samples = "
        f"{args.n_passes * len(per_sample_inputs)} measurements",
        flush=True,
    )

    for _p in range(args.n_passes):
        for i, entry in enumerate(per_sample_inputs):
            # Do not include allocator leftovers from the previous sample in
            # the incremental CUDA peak. Model weights remain resident and are
            # intentionally part of absolute peak_allocated.
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cuda_before = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()

            if prof is not None:
                prof.start_profile()

            with torch.inference_mode():
                out = model(**entry["inputs"], use_cache=True)

            torch.cuda.synchronize()
            cuda_peak = int(torch.cuda.max_memory_allocated())
            cuda_peak_delta = max(cuda_peak - cuda_before, 0)

            if prof is not None:
                prof.stop_profile()
                per_sample_records[i]["tflops"].append(
                    prof.get_total_flops() / 1e12
                )
                prof.reset_profile()

            n_tok, n_bytes, per_layer = _walk_kv(out)
            snap = _snapshot_stats(model)

            rec = per_sample_records[i]
            rec["kv_cache_tokens"].append(int(n_tok))
            rec["kv_cache_bytes"].append(int(n_bytes))
            rec["ltc_snapshot_peak_count"].append(snap["peak_count"])
            rec["ltc_snapshot_peak_bytes"].append(snap["peak_bytes"])
            rec["ltc_snapshot_current_count"].append(snap["current_count"])
            rec["ltc_snapshot_current_bytes"].append(snap["current_bytes"])
            rec["cuda_allocated_before_bytes"].append(cuda_before)
            rec["cuda_peak_allocated_bytes"].append(cuda_peak)
            rec["cuda_peak_delta_bytes"].append(cuda_peak_delta)
            rec["per_layer_kv_seq_len"] = per_layer

            del out
            torch.cuda.empty_cache()

    if prof is not None:
        prof.end_profile()

    # Per-sample medians.
    for rec in per_sample_records:
        if rec["tflops"]:
            rec["tflops_median"] = float(_median(rec["tflops"]))

        rec["kv_cache_tokens_median"] = int(_median(rec["kv_cache_tokens"]))
        rec["kv_cache_MB_median"] = float(_median(rec["kv_cache_bytes"])) / 2**20
        rec["ltc_snapshot_peak_count_median"] = int(
            _median(rec["ltc_snapshot_peak_count"])
        )
        rec["ltc_snapshot_peak_MB_median"] = (
            float(_median(rec["ltc_snapshot_peak_bytes"])) / 2**20
        )
        rec["kv_plus_ltc_snapshot_MB_median"] = (
            float(_median(rec["kv_cache_bytes"]))
            + float(_median(rec["ltc_snapshot_peak_bytes"]))
        ) / 2**20
        rec["cuda_peak_allocated_MB_median"] = (
            float(_median(rec["cuda_peak_allocated_bytes"])) / 2**20
        )
        rec["cuda_peak_delta_MB_median"] = (
            float(_median(rec["cuda_peak_delta_bytes"])) / 2**20
        )

    all_tflops = [t for rec in per_sample_records for t in rec["tflops"]]
    all_kv_tok = [
        t for rec in per_sample_records for t in rec["kv_cache_tokens"]
    ]
    all_kv_bytes = [
        b for rec in per_sample_records for b in rec["kv_cache_bytes"]
    ]
    all_snapshot_peak_count = [
        n
        for rec in per_sample_records
        for n in rec["ltc_snapshot_peak_count"]
    ]
    all_snapshot_peak_bytes = [
        b
        for rec in per_sample_records
        for b in rec["ltc_snapshot_peak_bytes"]
    ]
    all_cuda_peak = [
        b
        for rec in per_sample_records
        for b in rec["cuda_peak_allocated_bytes"]
    ]
    all_cuda_delta = [
        b for rec in per_sample_records for b in rec["cuda_peak_delta_bytes"]
    ]

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    mean_kv_bytes = _mean(all_kv_bytes)
    mean_snapshot_bytes = _mean(all_snapshot_peak_bytes)

    aggregate = {
        "prefill_tflops_mean": _mean(all_tflops),
        "prefill_kv_cache_tokens_mean": _mean(all_kv_tok),
        "prefill_kv_cache_MB_mean": (
            mean_kv_bytes / 2**20 if mean_kv_bytes is not None else None
        ),
        "ltc_snapshot_peak_count_mean": _mean(all_snapshot_peak_count),
        "ltc_snapshot_peak_MB_mean": (
            mean_snapshot_bytes / 2**20
            if mean_snapshot_bytes is not None
            else 0.0
        ),
        # This is a footprint sum (final prefill KV + peak LTC history), not
        # a claim that the two independent peaks occur at the same instant.
        "kv_plus_ltc_snapshot_MB_mean": (
            (mean_kv_bytes + mean_snapshot_bytes) / 2**20
            if mean_kv_bytes is not None and mean_snapshot_bytes is not None
            else None
        ),
        "cuda_peak_allocated_MB_mean": (
            _mean(all_cuda_peak) / 2**20 if all_cuda_peak else None
        ),
        "cuda_peak_delta_MB_mean": (
            _mean(all_cuda_delta) / 2**20 if all_cuda_delta else None
        ),
    }

    routing_ctx = getattr(model, "_routing_ctx", None)
    snapshot_policy = (
        str(getattr(routing_ctx, "correction_snapshot_policy", "none"))
        if routing_ctx is not None
        else "none"
    )
    snapshot_keep = (
        int(getattr(routing_ctx, "correction_snapshot_keep", 0))
        if routing_ctx is not None
        else 0
    )

    summary = {
        "config": args.config_name,
        "scope": "prefill_only",
        "bench_data": {
            "source": "COCO val 2014 (via lmms-lab/POPE images)",
            "prompt": "Describe this image in detail.",
            "cohort_size": len(per_sample_inputs),
            "cohort_dir": str(bench_dir),
        },
        "settings": {
            "n_warmup_passes": args.n_warmup_passes,
            "n_passes": args.n_passes,
            "deepspeed_available": deepspeed_available,
            "snapshot_policy": snapshot_policy,
            "snapshot_keep": snapshot_keep,
        },
        "per_sample": per_sample_records,
        "aggregate": aggregate,
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] wrote {out_p}  (scope: prefill_only)", flush=True)
    print(
        f"  snapshot policy                  : {snapshot_policy}"
        + (f" (keep={snapshot_keep})" if snapshot_policy == "fifo" else ""),
        flush=True,
    )
    if aggregate["prefill_tflops_mean"] is not None:
        print(
            f"  prefill TFLOPs/sample (mean)     : "
            f"{aggregate['prefill_tflops_mean']:.3f}",
            flush=True,
        )
    print(
        f"  prefill KV cache tokens (mean)   : "
        f"{aggregate['prefill_kv_cache_tokens_mean']:.0f}",
        flush=True,
    )
    print(
        f"  prefill KV cache MB (mean)       : "
        f"{aggregate['prefill_kv_cache_MB_mean']:.2f}",
        flush=True,
    )
    print(
        f"  LTC snapshot peak count (mean)   : "
        f"{aggregate['ltc_snapshot_peak_count_mean']:.2f}",
        flush=True,
    )
    print(
        f"  LTC snapshot peak MB (mean)      : "
        f"{aggregate['ltc_snapshot_peak_MB_mean']:.2f}",
        flush=True,
    )
    print(
        f"  KV + LTC snapshot MB (mean)      : "
        f"{aggregate['kv_plus_ltc_snapshot_MB_mean']:.2f}",
        flush=True,
    )
    print(
        f"  CUDA peak allocated MB (mean)    : "
        f"{aggregate['cuda_peak_allocated_MB_mean']:.2f}",
        flush=True,
    )
    print(
        f"  CUDA incremental peak MB (mean)  : "
        f"{aggregate['cuda_peak_delta_MB_mean']:.2f}",
        flush=True,
    )

    for rec in per_sample_records:
        per_layer = rec["per_layer_kv_seq_len"]
        print(
            f"  sample_{rec['sample_id']} ({len(per_layer)} layers): {per_layer}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
