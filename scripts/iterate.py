"""Fast iteration runner for Hyper-LUT Net research rounds.

Usage:
    python scripts/iterate.py --round N --tag "description-of-change"

Trains on a fixed small MNIST subset (5000 train, 1000 test) for 5 epochs
and appends results to results/iterations.csv. Designed for ~30-60s per run.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
torch.set_num_threads(2)

from torch.utils.data import Subset, DataLoader

from bitforge.data.datasets import build_dataset, DATASET_INFO
from bitforge.models.hyper_lut import HyperLUTNet
from bitforge.training import Trainer
from bitforge.utils.seed import set_seed


# Fixed iteration config — DO NOT change between rounds (that's hyperparameter tuning)
ITER_TRAIN_N = 5000
ITER_TEST_N = 1000
ITER_EPOCHS = 5
ITER_BATCH = 128
ITER_LR = 1e-3
ITER_SEED = 0
DATA_ROOT = "./data"
RESULTS_CSV = os.path.join(HERE, "..", "results", "iterations.csv")


def run_round(round_num: int, tag: str, model_kwargs: dict | None = None) -> dict:
    """Run one iteration round. Returns a results dict."""
    set_seed(ITER_SEED)
    info = DATASET_INFO["mnist"]

    # Cache datasets on first run
    train_ds = build_dataset("mnist", root=DATA_ROOT, train=True, augment="none")
    test_ds = build_dataset("mnist", root=DATA_ROOT, train=False, augment="none")
    train_subset = Subset(train_ds, list(range(ITER_TRAIN_N)))
    test_subset = Subset(test_ds, list(range(ITER_TEST_N)))
    train_loader = DataLoader(train_subset, batch_size=ITER_BATCH, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=ITER_BATCH, shuffle=False)

    # Default model kwargs (the "round 0" baseline) — can be overridden
    defaults = dict(
        in_channels=info.in_channels,
        num_classes=info.num_classes,
        img_size=info.img_size,
        hv_dim=512,
        k=4,
        num_luts=16,
        num_blocks=2,
        encoder_seed=42,
        binarize_input=False,
    )
    if model_kwargs:
        defaults.update(model_kwargs)
    model = HyperLUTNet(**defaults)
    n_params = sum(p.numel() for p in model.parameters())

    optim = torch.optim.Adam(model.parameters(), lr=ITER_LR)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        device="cpu",
        exp_name=f"iter.r{round_num}",
        seed=ITER_SEED,
        output_dir="./results",
        log_every=20,
        num_threads=2,
    )

    t0 = time.time()
    try:
        result = trainer.fit(epochs=ITER_EPOCHS)
        dt = time.time() - t0
        row = {
            "round": round_num,
            "tag": tag,
            "params": n_params,
            "best_top1": result.best.get("top1", 0.0),
            "final_top1": result.final.get("top1", 0.0),
            "final_loss": result.final.get("loss", 0.0),
            "time_s": round(dt, 1),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        dt = time.time() - t0
        row = {
            "round": round_num,
            "tag": tag,
            "params": n_params,
            "best_top1": 0.0,
            "final_top1": 0.0,
            "final_loss": 0.0,
            "time_s": round(dt, 1),
            "status": "fail",
            "error": str(e)[:200],
        }
        traceback.print_exc()
    return row


def append_csv(row: dict) -> None:
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--hv-dim", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--num-luts", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    args = p.parse_args()

    mk = {}
    if args.hv_dim is not None: mk["hv_dim"] = args.hv_dim
    if args.k is not None: mk["k"] = args.k
    if args.num_luts is not None: mk["num_luts"] = args.num_luts
    if args.num_blocks is not None: mk["num_blocks"] = args.num_blocks

    print(f"\n=== ROUND {args.round}: {args.tag} ===")
    row = run_round(args.round, args.tag, model_kwargs=mk or None)
    append_csv(row)
    print(f"\n=== RESULT: top1={row['best_top1']:.4f} loss={row['final_loss']:.4f} "
          f"time={row['time_s']}s status={row['status']} ===")
    if row["status"] != "ok":
        print(f"ERROR: {row['error']}")


if __name__ == "__main__":
    main()
