"""Run LLaVA-1.5-7B evaluation with CatchUp (Q-VER + LTC).

Loads the pretrained model via lmms-eval, optionally patches the decoder
for routing, evaluates the requested benchmark, and writes JSON metrics
and per-sample outputs under the Hydra run directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path (before hydra changes cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import os
from datetime import datetime

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from models.router import QVERouter
from models.dispatcher import TokenDispatcher
from models.patching import patch_model_for_routing, get_visual_token_finder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_router(cfg: DictConfig):
    """Create CatchUp routing, or skip routing for dense inference."""
    if cfg.routing.method == "none":
        return None
    if cfg.routing.method != "qver":
        raise ValueError(f"Unsupported routing method: {cfg.routing.method}")
    return QVERouter(
        drop_layers=list(cfg.routing.drop_layers),
        keep_ratios=list(cfg.routing.keep_ratios),
        monotonic=cfg.routing.get("monotonic", False),
    )


def run_lmms_eval(cfg: DictConfig) -> dict:
    """Run lmms-eval with the patched model.

    Uses lmms-eval's Python API to load a model, patch it, and evaluate.
    """
    from lmms_eval import evaluator
    from lmms_eval.models import get_model

    model_name = cfg.model.pretrained
    model_family = "llava"
    lmms_model_type = cfg.model.lmms_model_type
    if lmms_model_type != "llava_hf":
        raise ValueError("This release supports only LLaVA-1.5.")

    logger.info(f"Loading model: {model_name} (type: {lmms_model_type})")

    # Build model_args string for lmms-eval
    model_args = f"pretrained={model_name},device_map=auto"
    if cfg.model.get("attn_implementation"):
        model_args += f",attn_implementation={cfg.model.attn_implementation}"
    if cfg.model.get("dtype"):
        model_args += f",dtype={cfg.model.dtype}"
    if cfg.model.get("max_pixels"):
        model_args += f",max_pixels={cfg.model.max_pixels}"
    if cfg.model.get("enable_thinking") is not None:
        model_args += f",enable_thinking={cfg.model.enable_thinking}"

    # Load model via lmms-eval's standard API
    model_cls = get_model(lmms_model_type)
    lm = model_cls.create_from_arg_string(
        model_args,
        {"batch_size": cfg.eval.batch_size},
    )

    router = build_router(cfg)
    if router is not None:
        logger.info(f"Patching LLaVA with CatchUp: {cfg.routing.action}")
        ctx = patch_model_for_routing(
            lm._model, router=router, dispatcher=TokenDispatcher(),
            action=cfg.routing.action, model_family="llava",
        )
        lm._routing_ctx = ctx
    else:
        logger.info("No routing: dense LLaVA baseline")

    # Original LLaVA-1.5 RefCOCO prompt / bbox conventions.
    prompt_style = (os.environ.get("REFCOCO_PROMPT_STYLE")
                    or cfg.eval.get("prompt_style", None) or "full")
    bbox_format = os.environ.get("BBOX_COORD_FORMAT") or "normalized"
    os.environ["REFCOCO_PROMPT_STYLE"] = prompt_style
    os.environ["BBOX_COORD_FORMAT"] = bbox_format
    logger.info(f"RefCOCO preset: bbox={bbox_format}, prompt={prompt_style}")

    # Run evaluation
    tasks = list(cfg.eval.benchmarks)
    logger.info(f"Evaluating on: {tasks}")

    eval_kwargs = dict(
        model=lm,
        tasks=tasks,
        batch_size=cfg.eval.batch_size,
        log_samples=cfg.eval.get("log_samples", False),
    )
    limit = cfg.eval.get("limit", None)
    if limit is not None:
        eval_kwargs["limit"] = limit

    results = evaluator.simple_evaluate(**eval_kwargs)

    return results


def log_results(cfg: DictConfig, results: dict) -> None:
    """Log results to console and local file."""
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    result_summary = {
        "model": cfg.model.pretrained,
        "routing_method": cfg.routing.method,
        "routing_action": cfg.routing.get("action", "none"),
        "keep_ratio": cfg.routing.get("keep_ratio", None) or list(cfg.routing.get("keep_ratios", [1.0])),
        "timestamp": datetime.now().isoformat(),
    }

    if "results" in results:
        for task_name, task_results in results["results"].items():
            logger.info(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
                    result_summary[f"{task_name}/{metric}"] = value

    # Per-experiment artifact dir = Hydra's runtime output_dir.
    # Hydra 1.2+ default is chdir=false, so relative Path("results.json")
    # would land in launch-CWD (project root) and get clobbered across arms.
    # Resolve against HydraConfig explicitly → each arm's files stay isolated.
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        run_dir = Path.cwd()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Annotate the summary with run_dir so the global JSONL can be back-traced.
    result_summary["run_dir"] = str(run_dir)

    local_results = run_dir / "results.json"
    with open(local_results, "w") as f:
        json.dump(result_summary, f, indent=2)
    logger.info(f"Results saved to {local_results}")

    global_results = _PROJECT_ROOT / "experiments" / "eval_results.jsonl"
    with open(global_results, "a") as f:
        f.write(json.dumps(result_summary) + "\n")
    logger.info(f"Appended to {global_results}")

    # Per-sample outputs (model responses) if log_samples was enabled
    if "samples" in results:
        samples_file = run_dir / "samples.json"
        with open(samples_file, "w") as f:
            json.dump(results["samples"], f, indent=2, default=str)
        logger.info(f"Per-sample outputs saved to {samples_file}")


@hydra.main(config_path=str(_PROJECT_ROOT / "configs"), config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if OmegaConf.select(cfg, "hydra.verbose", default=False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    results = run_lmms_eval(cfg)
    log_results(cfg, results)


if __name__ == "__main__":
    main()
