"""Comparison: single binary expert vs single FP CNN of same architecture."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
torch.set_num_threads(2)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from bitforge.data.datasets import build_dataset, DATASET_INFO
from bitforge.training import Trainer
from bitforge.models.baselines.layers import BinaryConv2d, BinaryLinear
from bitforge.utils.seed import set_seed

ITER_TRAIN_N = 5000
ITER_TEST_N = 1000
ITER_EPOCHS = 5
ITER_BATCH = 128
ITER_LR = 1e-3
DATA_ROOT = "./data"


def make_cnn(in_ch, num_classes, bw, n_convs=6, binary=True, first_fp=True):
    """Make a CNN with N convs, optionally binary."""
    Conv = BinaryConv2d if binary else nn.Conv2d
    Lin = BinaryLinear if binary else nn.Linear
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            # First conv
            if first_fp:
                self.convs.append(nn.Conv2d(in_ch, bw, 3, 1, 1, bias=False))
            else:
                self.convs.append(Conv(in_ch, bw, 3, 1, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(bw))
            widths = [bw, bw*2, bw*4, bw*4, bw*4, bw*4, bw*4][:n_convs]
            for i in range(len(widths)-1):
                stride = 2 if i < 2 else 1
                self.convs.append(Conv(widths[i], widths[i+1], 3, stride, 1, bias=False))
                self.bns.append(nn.BatchNorm2d(widths[i+1]))
            self.fc = Lin(widths[-1], num_classes, bias=False)
        def forward(self, x):
            x = F.gelu(self.bns[0](self.convs[0](x)))
            for i in range(1, len(self.convs)):
                x = F.gelu(self.bns[i](self.convs[i](x)))
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)
    return CNN()


def run(round_num, tag, binary=True, first_fp=True, n_convs=6, bw=32):
    set_seed(0)
    train_ds = build_dataset("mnist", root=DATA_ROOT, train=True, augment="light")
    test_ds = build_dataset("mnist", root=DATA_ROOT, train=False, augment="none")
    train_loader = DataLoader(Subset(train_ds, list(range(ITER_TRAIN_N))), batch_size=ITER_BATCH, shuffle=True)
    test_loader = DataLoader(Subset(test_ds, list(range(ITER_TEST_N))), batch_size=ITER_BATCH, shuffle=False)
    model = make_cnn(1, 10, bw, n_convs=n_convs, binary=binary, first_fp=first_fp)
    n_params = sum(p.numel() for p in model.parameters())
    n_bits = sum(p.numel() * (1 if binary and "binary" in n.lower() else 32) for n, p in model.named_parameters())
    # For FP, all bits are 32; for binary with first_fp, the first conv is FP
    n_bits = 0
    for name, p in model.named_parameters():
        if binary:
            if first_fp and "convs.0" in name:
                n_bits += p.numel() * 32  # FP first conv
            else:
                n_bits += p.numel() * 1   # binary
        else:
            n_bits += p.numel() * 32
    optim = torch.optim.Adam(model.parameters(), lr=ITER_LR)
    trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader,
                      optimizer=optim, device="cpu", exp_name=f"abl.r{round_num}",
                      seed=0, output_dir="./results", log_every=100, num_threads=2)
    t0 = time.time()
    result = trainer.fit(epochs=ITER_EPOCHS)
    dt = time.time() - t0
    row = {"round": round_num, "tag": tag, "ablation": "fp_vs_binary",
           "params": n_params, "bits": n_bits,
           "best_top1": result.best.get("top1", 0.0),
           "final_top1": result.final.get("top1", 0.0),
           "final_loss": result.final.get("loss", 0.0),
           "time_s": round(dt, 1), "status": "ok", "error": ""}
    csv_path = os.path.join(HERE, "..", "results", "ablations.csv")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)
    print(f"\n=== R{round_num} {tag}: top1={row['best_top1']:.4f} params={n_params:,} bits={n_bits:,} time={row['time_s']}s ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--binary", type=int, default=1)
    p.add_argument("--first-fp", type=int, default=1)
    p.add_argument("--n-convs", type=int, default=6)
    p.add_argument("--bw", type=int, default=32)
    args = p.parse_args()
    run(args.round, args.tag, binary=bool(args.binary), first_fp=bool(args.first_fp),
        n_convs=args.n_convs, bw=args.bw)
