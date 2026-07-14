"""Run a full benchmark sweep: all baselines + Hyper-LUT, across datasets, with
configurable seeds. Produces a Markdown report at results/report.md.

Designed to be runnable in chunks — each call to run_one is independent and
serializes its result to disk, so you can resume after interruptions by
skipping experiments whose summary JSON already exists.

Examples:
    # Quick smoke sweep on MNIST, 1 seed, 1 epoch each
    python experiments/run_sweep.py --datasets mnist --seeds 0 --epochs 1 --tag smoke

    # Full CIFAR-10 sweep with 3 seeds
    python experiments/run_sweep.py --datasets cifar10 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitforge.utils.config import load_config
from bitforge.utils.logging import get_logger, write_json
from bitforge.evaluation.runner import run_multi_seed, build_exp_name
from bitforge.evaluation.reporter import Reporter


ALL_MODELS = ["fp_resnet", "xnor_net", "bireal", "irnet", "reactnet", "hyper_lut"]


def parse_args():
    p = argparse.ArgumentParser(description="Run a BitForge benchmark sweep")
    p.add_argument("--datasets", type=str, default="mnist,cifar10,cifar100",
                   help="Comma-separated dataset names")
    p.add_argument("--models", type=str, default=",".join(ALL_MODELS),
                   help="Comma-separated model names")
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="Comma-separated seeds")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--time-budget", type=float, default=None,
                   help="Per-run wall-clock budget in seconds")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip experiments whose summary JSON exists")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def build_model_config(model_name: str, dataset_name: str, args) -> "Config":
    """Load the default config for the dataset, then override the model name + params."""
    cfg_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "configs", dataset_name, "default.yaml"
    ))
    overrides = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.time_budget is not None:
        overrides["runtime.time_budget_s"] = args.time_budget
    if args.tag:
        overrides["runtime.tag"] = args.tag
    cfg = load_config(cfg_path, overrides=overrides if overrides else None)
    cfg.set("model.name", model_name)
    # baseline-specific defaults
    if model_name == "fp_resnet":
        cfg.set("model.base_width", 16)
    elif model_name in ("xnor_net", "bireal", "irnet", "reactnet"):
        cfg.set("model.base_width", 32)
    # hyper_lut uses defaults from config file
    return cfg


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    logger = get_logger("sweep")

    summaries = []
    for ds in datasets:
        for mdl in models:
            cfg = build_model_config(mdl, ds, args)
            exp_name = build_exp_name(mdl, ds, args.tag)
            summary_path = os.path.join(args.output_dir, "metrics", f"{exp_name}.summary.json")
            if args.skip_existing and os.path.exists(summary_path):
                logger.info(f"[skip] {exp_name} (summary exists at {summary_path})")
                from bitforge.utils.logging import read_json
                summaries.append(read_json(summary_path))
                continue

            logger.info(f"=== running {exp_name} (seeds={seeds}) ===")
            try:
                summary = run_multi_seed(cfg, seeds=seeds, output_dir=args.output_dir)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"[failed] {exp_name}: {e}")
                import traceback; traceback.print_exc()
                continue

    # write report
    reporter = Reporter(output_dir=args.output_dir)
    report_path = reporter.write_markdown(summaries, filename="report.md")
    logger.info(f"Report written to {report_path}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
