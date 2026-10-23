#!/usr/bin/env python3
"""Runtime microbenchmark on the three-image bundled cohort.

Run via: bash scripts/run_efficiency.sh --mode runtime --budget 64 --gpu 0
Direct entrypoint: python profiler/bench_runtime.py --config-name
experiment/llava15/avg64/llava15_qver_avg64 --out experiments/runtime/catchup_avg64.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_RELEASE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RELEASE_ROOT))

# Bundled cohort directory.
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
        img_p = BENCH_BUNDLED / s["image"]
        if not img_p.exists():
            raise FileNotFoundError(f"Manifest references missing image: {img_p}")
    return BENCH_BUNDLED, manifest


# ─────────────────────────────────────────────────────────────────────
# Model + routing build (mirrors profiler/measure_metrics.py loader)
# ─────────────────────────────────────────────────────────────────────
def _build_model_and_apply_routing(cfg_path: Path):
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from models.router import QVERouter
    from models.dispatcher import TokenDispatcher
    from models.patching import patch_model_for_routing

    cfg = OmegaConf.load(cfg_path)

    model_name = "llava15_7b_hf"
    for d in cfg.get("defaults", []):
        if hasattr(d, "keys") and "/model" in d:
            model_name = str(d["/model"])
            break
    model_cfg = OmegaConf.load(_RELEASE_ROOT / "configs" / "model" / f"{model_name}.yaml")

    pretrained = model_cfg.pretrained
    if model_cfg.get("lmms_model_type", "llava_hf") != "llava_hf":
        raise ValueError("Only LLaVA-1.5 is supported.")
    from transformers import LlavaForConditionalGeneration as ModelCls
    model_family = "llava"
    dtype = torch.float16
    print(f"[CatchUp] loading {pretrained} (dtype={dtype})", flush=True)
    load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model = ModelCls.from_pretrained(pretrained, **load_kwargs).to("cuda").eval()

    proc_kwargs = {}
    if model_cfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = int(model_cfg.max_pixels)
    processor = AutoProcessor.from_pretrained(pretrained, **proc_kwargs)

    r_cfg = cfg.routing
    if r_cfg.method == "none":
        print("[bench] dense LLaVA baseline", flush=True)
    elif r_cfg.method == "qver":
        router = QVERouter(
            drop_layers=list(r_cfg.drop_layers),
            keep_ratios=list(r_cfg.keep_ratios),
            monotonic=r_cfg.get("monotonic", False),
        )
        patch_model_for_routing(
            model, router, TokenDispatcher(), action=r_cfg.action,
            model_family=model_family,
        )
    else:
        raise ValueError(f"Unsupported routing method: {r_cfg.method}")

    return model, processor, model_family


def _build_inputs_for_sample(processor, model, sample: dict, bench_dir: Path):
    img = Image.open(bench_dir / sample["image"]).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": sample["question"]},
    ]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device).long()
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────
# Timing helpers (CUDA Events)
# ─────────────────────────────────────────────────────────────────────
def _gpu_time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _summary(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50_median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True,
                    help="release experiment config path, e.g. experiment/llava15/avg192/...")
    ap.add_argument("--n-passes", type=int, default=3,
                    help="number of timed passes through the three-image cohort")
    ap.add_argument("--n-warmup-passes", type=int, default=1,
                    help="warmup passes through cohort (not timed, primes cuDNN/caches)")
    ap.add_argument("--n-decode-tokens", type=int, default=64,
                    help="forced decode steps per sample iteration")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = _RELEASE_ROOT / "configs" / f"{args.config_name}.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 1

    bench_dir, manifest = _load_bundled_cohort()
    print(f"[bench] cohort source: {bench_dir} ({len(manifest)} samples)", flush=True)
    model, processor, model_family = _build_model_and_apply_routing(cfg_path)

    # Pre-build inputs for each sample once (CPU-side preprocessing not timed)
    per_sample_inputs = []
    for s in manifest:
        inp = _build_inputs_for_sample(processor, model, s, bench_dir)
        per_sample_inputs.append({
            "sample": s,
            "inputs": inp,
            "prefill_tokens": int(inp["input_ids"].shape[1]),
        })

    print(f"[bench] cohort = {len(per_sample_inputs)} samples", flush=True)
    for entry in per_sample_inputs:
        print(f"  sample_{entry['sample']['sample_id']:>2}  "
              f"({entry['sample']['category']:<8})  "
              f"img={tuple(entry['sample']['image_WxH'])}  "
              f"prefill={entry['prefill_tokens']:>4} tokens", flush=True)

    gen_kwargs = dict(
        max_new_tokens=args.n_decode_tokens,
        min_new_tokens=args.n_decode_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )

    # ── Warmup passes (full pass through all samples, not timed) ──
    print(f"[bench] warmup {args.n_warmup_passes} pass × {len(per_sample_inputs)} samples ...", flush=True)
    for _ in range(args.n_warmup_passes):
        for entry in per_sample_inputs:
            with torch.inference_mode():
                _ = model.generate(**entry["inputs"], **gen_kwargs)
        torch.cuda.empty_cache()

    # ── Timed passes ──
    per_sample_records = []
    for entry in per_sample_inputs:
        per_sample_records.append({
            "sample_id": entry["sample"]["sample_id"],
            "category":  entry["sample"]["category"],
            "question":  entry["sample"]["question"],
            "image_WxH": entry["sample"]["image_WxH"],
            "prefill_tokens": entry["prefill_tokens"],
            "prefill_ms": [],
            "e2e_ms":     [],
            "response":   None,
        })

    print(f"[bench] timed {args.n_passes} passes × {len(per_sample_inputs)} samples = "
          f"{args.n_passes * len(per_sample_inputs)} measurements per metric", flush=True)
    for p in range(args.n_passes):
        for i, entry in enumerate(per_sample_inputs):
            captured = {"out": None}
            def _prefill_call():
                with torch.inference_mode():
                    _ = model(**entry["inputs"], use_cache=True)
            def _e2e_call():
                with torch.inference_mode():
                    captured["out"] = model.generate(**entry["inputs"], **gen_kwargs)
            per_sample_records[i]["prefill_ms"].append(_gpu_time_ms(_prefill_call))
            per_sample_records[i]["e2e_ms"].append(_gpu_time_ms(_e2e_call))

            # First-pass: decode and record the model response (do_sample=False
            # → deterministic, identical across passes, only need to capture once).
            if p == 0 and captured["out"] is not None:
                seqs = captured["out"].sequences
                pref_len = entry["prefill_tokens"]
                gen_ids = seqs[0, pref_len:].tolist()
                try:
                    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
                except AttributeError:
                    text = processor.decode(gen_ids, skip_special_tokens=True)
                per_sample_records[i]["response"] = text.strip()

    # Per-sample median + cohort aggregate
    for rec in per_sample_records:
        rec["prefill_ms_median"]  = float(np.median(rec["prefill_ms"]))
        rec["e2e_ms_median"]      = float(np.median(rec["e2e_ms"]))
        rec["decode_ms_per_tok_median"] = (rec["e2e_ms_median"] - rec["prefill_ms_median"]) / args.n_decode_tokens

    all_prefill = [t for rec in per_sample_records for t in rec["prefill_ms"]]
    all_e2e     = [t for rec in per_sample_records for t in rec["e2e_ms"]]
    all_decode  = [(e - p) / args.n_decode_tokens
                   for rec in per_sample_records
                   for p, e in zip(rec["prefill_ms"], rec["e2e_ms"])]
    all_tok_per_sec = [1000.0 / d if d > 0 else 0.0 for d in all_decode]

    cap = torch.cuda.get_device_capability(0)
    hw = {
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_capability": f"sm_{cap[0]}{cap[1]}",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "host": platform.node(),
    }

    summary = {
        "config": args.config_name,
        "hw": hw,
        "bench_data": {
            "source": "COCO val 2014 (via lmms-lab/POPE images)",
            "prompt": "Describe this image in detail.",
            "cohort_size": len(per_sample_inputs),
            "cohort_dir": str(bench_dir),
        },
        "settings": {
            "n_warmup_passes": args.n_warmup_passes,
            "n_passes": args.n_passes,
            "n_decode_tokens": args.n_decode_tokens,
        },
        "per_sample": per_sample_records,
        "aggregate": {
            "prefill_ms":         _summary(all_prefill),
            "e2e_ms":             _summary(all_e2e),
            "decode_ms_per_tok":  _summary(all_decode),
            "decode_tokens_per_sec": _summary(all_tok_per_sec),
        },
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w") as f:
        json.dump(summary, f, indent=2)

    a = summary["aggregate"]
    print(f"\n[bench] {args.config_name}  on  {hw['gpu_name']}")
    print(f"  prefill_ms          median={a['prefill_ms']['p50_median']:.2f}  p10={a['prefill_ms']['p10']:.2f}  p90={a['prefill_ms']['p90']:.2f}")
    print(f"  e2e_ms              median={a['e2e_ms']['p50_median']:.2f}  p10={a['e2e_ms']['p10']:.2f}  p90={a['e2e_ms']['p90']:.2f}")
    print(f"  decode_ms_per_tok   median={a['decode_ms_per_tok']['p50_median']:.3f}")
    print(f"  decode_tok_per_sec  median={a['decode_tokens_per_sec']['p50_median']:.1f}")
    # Show first 2 sample responses for quick sanity check
    print(f"\n  --- sanity (first 2 sample responses) ---")
    for rec in per_sample_records[:2]:
        resp = (rec["response"] or "")[:160].replace("\n", " ")
        print(f"  sample_{rec['sample_id']:>2} ({rec['category']}): {resp}")
    print(f"[OK] wrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
