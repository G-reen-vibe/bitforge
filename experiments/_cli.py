"""Common CLI for experiment scripts.

Usage:
    python experiments/run_baseline.py --model reactnet --dataset cifar10 --seeds 0,1,2
    python experiments/run_hyper_lut.py --dataset mnist --seeds 0 --epochs 1 --tag smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitforge.utils.config import Config, load_config


def parse_common_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
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
    return p.parse_args()


def build_config_from_args(args, default_config_path: str = None) -> Config:
    """Load config (from --config or default for dataset) + apply CLI overrides."""
    if args.config:
        cfg_path = args.config
    elif default_config_path:
        cfg_path = default_config_path.format(dataset=args.dataset)
    else:
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", args.dataset, "default.yaml"
        )
    cfg_path = os.path.abspath(cfg_path)
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
    return load_config(cfg_path, overrides=overrides if overrides else None)


def parse_seeds(seeds_str: str) -> List[int]:
    return [int(s) for s in seeds_str.split(",") if s.strip()]
