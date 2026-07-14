"""Run a Hyper-LUT Net experiment.

Examples:
    # Smoke test (1 epoch, MNIST, single seed)
    python experiments/run_hyper_lut.py --dataset mnist --seeds 0 --epochs 1 --tag smoke

    # Multi-seed CIFAR-10
    python experiments/run_hyper_lut.py --dataset cifar10 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Run a Hyper-LUT Net experiment")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (defaults to configs/<dataset>/default.yaml)")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["mnist", "cifar10", "cifar100"])
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated seeds, e.g. '0,1,2'")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--hv-dim", type=int, default=None, help="Hyper-LUT hypervector dim D")
    p.add_argument("--k", type=int, default=None, help="LUT input bits")
    p.add_argument("--num-luts", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
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
    if args.hv_dim is not None:
        overrides["model.hv_dim"] = args.hv_dim
    if args.k is not None:
        overrides["model.k"] = args.k
    if args.num_luts is not None:
        overrides["model.num_luts"] = args.num_luts
    if args.num_blocks is not None:
        overrides["model.num_blocks"] = args.num_blocks
    cfg = load_config(cfg_path, overrides=overrides if overrides else None)
    cfg.set("model.name", "hyper_lut")
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
