"""Run a baseline experiment: FP-ResNet / XNOR-Net / Bi-Real / IR-Net / ReActNet.

Examples:
    # Smoke test (1 epoch, MNIST, single seed)
    python experiments/run_baseline.py --model reactnet --dataset mnist --seeds 0 --epochs 1 --tag smoke

    # Multi-seed CIFAR-10
    python experiments/run_baseline.py --model reactnet --dataset cifar10 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Run a BitForge baseline experiment")
    p.add_argument("--model", type=str, default="fp_resnet",
                   choices=["fp_resnet", "xnor_net", "bireal", "irnet", "reactnet"])
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (defaults to configs/<dataset>/default.yaml)")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["mnist", "cifar10", "cifar100"])
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated seeds, e.g. '0,1,2'")
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    p.add_argument("--lr", type=float, default=None, help="Override train.lr")
    p.add_argument("--tag", type=str, default="", help="Tag for experiment name")
    p.add_argument("--time-budget", type=float, default=None,
                   help="Override runtime.time_budget_s (per run)")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--width-multiplier", type=float, default=None)
    p.add_argument("--blocks-per-stage", type=int, default=None)
    return p.parse_args()


def build_config_from_args(args):
    from bitforge.utils.config import load_config
    if args.config:
        cfg_path = args.config
    else:
        cfg_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "configs", args.dataset, "default.yaml"
        ))
    overrides = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["data.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["train.lr"] = args.lr
    if args.tag:
        overrides["runtime.tag"] = args.tag
    if args.time_budget is not None:
        overrides["runtime.time_budget_s"] = args.time_budget
    if args.num_threads is not None:
        overrides["runtime.num_threads"] = args.num_threads
    if args.width_multiplier is not None:
        overrides["model.width_multiplier"] = args.width_multiplier
    if args.blocks_per_stage is not None:
        overrides["model.blocks_per_stage"] = args.blocks_per_stage
    cfg = load_config(cfg_path, overrides=overrides if overrides else None)
    # always set the model name
    cfg.set("model.name", args.model)
    # baseline defaults (if not in config)
    if cfg.model.get("width_multiplier") is None:
        cfg.set("model.width_multiplier", 1.0)
    if cfg.model.get("blocks_per_stage") is None:
        cfg.set("model.blocks_per_stage", 2)
    if cfg.model.get("base_width") is None:
        cfg.set("model.base_width", 16 if args.model == "fp_resnet" else 32)
    return cfg


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = build_config_from_args(args)
    from bitforge.evaluation.runner import run_multi_seed
    summary = run_multi_seed(cfg, seeds=seeds, output_dir=args.output_dir)
    print("\n=== Summary ===")
    agg = summary["aggregated"]
    print(f"Best top1:   {agg.get('best.top1', 'N/A')}")
    print(f"Final top1:  {agg.get('final.top1', 'N/A')}")


if __name__ == "__main__":
    main()
