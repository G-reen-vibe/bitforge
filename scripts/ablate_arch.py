"""Single-expert architecture ablations — fast (~45s each).

Since R1 showed single expert gets top1=0.930 and the MoE only adds +0.026,
the expert architecture is the dominant factor. These ablations isolate
each architectural choice in the expert.
"""
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
ITER_SEED = 0
DATA_ROOT = "./data"
RESULTS_CSV = os.path.join(HERE, "..", "results", "ablations.csv")


def get_loaders():
    train_ds = build_dataset("mnist", root=DATA_ROOT, train=True, augment="light")
    test_ds = build_dataset("mnist", root=DATA_ROOT, train=False, augment="none")
    train_loader = DataLoader(Subset(train_ds, list(range(ITER_TRAIN_N))),
                              batch_size=ITER_BATCH, shuffle=True)
    test_loader = DataLoader(Subset(test_ds, list(range(ITER_TEST_N))),
                             batch_size=ITER_BATCH, shuffle=False)
    return train_loader, test_loader


def make_expert(in_ch, num_classes, bw, n_binary_convs=4,
                first_fp=True, activation="gelu", kernel_size=3):
    """Build a configurable binary expert."""
    layers = []
    # First conv
    if first_fp:
        layers += [nn.Conv2d(in_ch, bw, kernel_size, 1, kernel_size//2, bias=False),
                   nn.BatchNorm2d(bw)]
    else:
        layers += [BinaryConv2d(in_ch, bw, kernel_size, 1, kernel_size//2, bias=False),
                   nn.BatchNorm2d(bw)]
    act = F.gelu if activation == "gelu" else F.relu
    # We'll build a custom module since activation needs to be applied
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            # first conv
            if first_fp:
                self.convs.append(nn.Conv2d(in_ch, bw, kernel_size, 1, kernel_size//2, bias=False))
            else:
                self.convs.append(BinaryConv2d(in_ch, bw, kernel_size, 1, kernel_size//2, bias=False))
            self.bns.append(nn.BatchNorm2d(bw))
            # binary convs
            widths = [bw, bw*2, bw*4, bw*4, bw*4, bw*4][:n_binary_convs]
            for i in range(len(widths)-1):
                stride = 2 if i == 0 else (2 if i == 1 else 1)  # downsample at conv2, conv3
                self.convs.append(BinaryConv2d(widths[i], widths[i+1], kernel_size, stride, kernel_size//2, bias=False))
                self.bns.append(nn.BatchNorm2d(widths[i+1]))
            self.fc = BinaryLinear(widths[-1], num_classes, bias=False)
            self.act = act
        def forward(self, x):
            x = self.act(self.bns[0](self.convs[0](x)))
            for i in range(1, len(self.convs)):
                x = self.act(self.bns[i](self.convs[i](x)))
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)
    return Expert()


def run(round_num, tag, **expert_kwargs):
    set_seed(ITER_SEED)
    train_loader, test_loader = get_loaders()
    model = make_expert(1, 10, **expert_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    n_bits = 0
    for name, p in model.named_parameters():
        bw = 1 if "binary" in name.lower() else 32
        n_bits += p.numel() * bw
    optim = torch.optim.Adam(model.parameters(), lr=ITER_LR)
    trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader,
                      optimizer=optim, device="cpu", exp_name=f"abl.r{round_num}",
                      seed=ITER_SEED, output_dir="./results", log_every=50, num_threads=2)
    t0 = time.time()
    try:
        result = trainer.fit(epochs=ITER_EPOCHS)
        dt = time.time() - t0
        row = {"round": round_num, "tag": tag, "ablation": "single_arch",
               "params": n_params, "bits": n_bits,
               "best_top1": result.best.get("top1", 0.0),
               "final_top1": result.final.get("top1", 0.0),
               "final_loss": result.final.get("loss", 0.0),
               "time_s": round(dt, 1), "status": "ok", "error": ""}
    except Exception as e:
        dt = time.time() - t0
        row = {"round": round_num, "tag": tag, "ablation": "single_arch",
               "params": n_params, "bits": n_bits,
               "best_top1": 0.0, "final_top1": 0.0, "final_loss": 0.0,
               "time_s": round(dt, 1), "status": "fail", "error": str(e)[:200]}
        import traceback; traceback.print_exc()
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"\n=== R{round_num} {tag}: top1={row['best_top1']:.4f} params={n_params:,} time={row['time_s']}s ===")
    return row


# Default expert config (matches r74 best single-expert)
DEFAULTS = dict(bw=32, n_binary_convs=4, first_fp=True, activation="gelu", kernel_size=3)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--bw", type=int, default=32)
    p.add_argument("--n-binary-convs", type=int, default=4)
    p.add_argument("--first-fp", type=int, default=1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--kernel-size", type=int, default=3)
    args = p.parse_args()
    run(args.round, args.tag, bw=args.bw, n_binary_convs=args.n_binary_convs,
        first_fp=bool(args.first_fp), activation=args.activation,
        kernel_size=args.kernel_size)
