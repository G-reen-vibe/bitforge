"""Ablation runner for BinaryMoE — round-by-round analysis.

Each ablation isolates ONE variable to understand its contribution.
All ablations use the same train/test subset (5K MNIST, 5 epochs) for fair comparison.

Usage:
    python scripts/ablate.py --round 1 --tag "single-expert-no-moe" --ablation single_expert
    python scripts/ablate.py --round 2 --tag "no-gate-uniform" --ablation no_gate
    ...
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
from bitforge.training import Trainer
from bitforge.models.binary_moe import BinaryMoE
from bitforge.models.binary_moe.expert import TinyBinaryExpert
from bitforge.utils.seed import set_seed
import torch.nn as nn
import torch.nn.functional as F

# Fixed iteration config (matches iterate.py)
ITER_TRAIN_N = 5000
ITER_TEST_N = 1000
ITER_EPOCHS = 5
ITER_BATCH = 128
ITER_LR = 1e-3
ITER_SEED = 0
DATA_ROOT = "./data"
RESULTS_CSV = os.path.join(HERE, "..", "results", "ablations.csv")


# Best BinaryMoE config from r74
BEST_CONFIG = dict(
    in_channels=1,
    img_size=28,
    num_classes=10,
    num_experts=4,
    topk=2,
    base_width=32,
    gate_noise=1.0,
)


def get_loaders():
    train_ds = build_dataset("mnist", root=DATA_ROOT, train=True, augment="light")
    test_ds = build_dataset("mnist", root=DATA_ROOT, train=False, augment="none")
    train_loader = DataLoader(Subset(train_ds, list(range(ITER_TRAIN_N))),
                              batch_size=ITER_BATCH, shuffle=True)
    test_loader = DataLoader(Subset(test_ds, list(range(ITER_TEST_N))),
                             batch_size=ITER_BATCH, shuffle=False)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Ablation models
# -----------------------------------------------------------------------------

class SingleExpertModel(nn.Module):
    """Ablation 1: just one expert, no MoE."""
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**BEST_CONFIG, **kwargs}
        self.expert = TinyBinaryExpert(cfg["in_channels"], cfg["num_classes"], cfg["base_width"])
    def forward(self, x):
        return self.expert(x)


class NoGateMoE(nn.Module):
    """Ablation 2: uniform top-2 weights (no learned gate)."""
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**BEST_CONFIG, **kwargs}
        self.num_experts = cfg["num_experts"]
        self.topk = cfg["topk"]
        # No gate — we'll use fixed uniform routing
        self.experts = nn.ModuleList([
            TinyBinaryExpert(cfg["in_channels"], cfg["num_classes"], cfg["base_width"])
            for _ in range(self.num_experts)
        ])
    def forward(self, x):
        # All experts active, uniform average (no gate)
        logits = torch.stack([e(x) for e in self.experts], dim=1)  # (N, E, C)
        return logits.mean(dim=1)  # uniform average


class RandomGateMoE(nn.Module):
    """Ablation 3: random gate (routing is random, not learned)."""
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**BEST_CONFIG, **kwargs}
        self.num_experts = cfg["num_experts"]
        self.topk = cfg["topk"]
        self.experts = nn.ModuleList([
            TinyBinaryExpert(cfg["in_channels"], cfg["num_classes"], cfg["base_width"])
            for _ in range(self.num_experts)
        ])
    def forward(self, x):
        N = x.size(0)
        # Random top-k routing per sample
        idx = torch.randperm(self.num_experts)[:self.topk]
        gates = torch.zeros(N, self.num_experts, device=x.device)
        gates[:, idx] = 1.0 / self.topk
        logits = torch.stack([e(x) for e in self.experts], dim=1)  # (N, E, C)
        return (gates.unsqueeze(-1) * logits).sum(dim=1)


class FPExpertMoE(nn.Module):
    """Ablation 5: FP experts (no binarization) — upper bound."""
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**BEST_CONFIG, **kwargs}
        from bitforge.models.binary_moe.gate import TopKGate
        self.gate = TopKGate(cfg["in_channels"], cfg["img_size"], cfg["num_experts"], cfg["topk"])
        self.gate.gate_noise = cfg["gate_noise"]
        # FP experts (replace BinaryConv2d with nn.Conv2d, BinaryLinear with nn.Linear)
        bw = cfg["base_width"]
        self.experts = nn.ModuleList([
            self._make_fp_expert(cfg["in_channels"], cfg["num_classes"], bw)
            for _ in range(cfg["num_experts"])
        ])
    def _make_fp_expert(self, in_ch, num_classes, bw):
        return nn.Sequential(
            nn.Conv2d(in_ch, bw, 3, 1, 1, bias=False), nn.BatchNorm2d(bw), nn.GELU(),
            nn.Conv2d(bw, bw*2, 3, 2, 1, bias=False), nn.BatchNorm2d(bw*2), nn.GELU(),
            nn.Conv2d(bw*2, bw*4, 3, 2, 1, bias=False), nn.BatchNorm2d(bw*4), nn.GELU(),
            nn.Conv2d(bw*4, bw*4, 3, 1, 1, bias=False), nn.BatchNorm2d(bw*4), nn.GELU(),
            nn.Conv2d(bw*4, bw*4, 3, 1, 1, bias=False), nn.BatchNorm2d(bw*4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(bw*4, num_classes, bias=False),
        )
    def forward(self, x):
        gates, _ = self.gate(x)
        logits = torch.stack([e(x) for e in self.experts], dim=1)
        return (gates.unsqueeze(-1) * logits).sum(dim=1)


class AllExpertsMoE(nn.Module):
    """Ablation 4: all experts active (topk=num_experts)."""
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**BEST_CONFIG, **kwargs}
        cfg["topk"] = cfg["num_experts"]  # all active
        self.model = BinaryMoE(**cfg)
    def forward(self, x):
        return self.model(x)


class AllBinaryExpertMoE(nn.Module):
    """Ablation 9: all binary convs (no FP first conv)."""
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**BEST_CONFIG, **kwargs}
        from bitforge.models.baselines.layers import BinaryConv2d, BinaryLinear
        from bitforge.models.binary_moe.gate import TopKGate
        self.gate = TopKGate(cfg["in_channels"], cfg["img_size"], cfg["num_experts"], cfg["topk"])
        self.gate.gate_noise = cfg["gate_noise"]
        bw = cfg["base_width"]
        self.experts = nn.ModuleList([
            self._make_all_binary_expert(cfg["in_channels"], cfg["num_classes"], bw)
            for _ in range(cfg["num_experts"])
        ])
    def _make_all_binary_expert(self, in_ch, num_classes, bw):
        return nn.Sequential(
            BinaryConv2d(in_ch, bw, 3, 1, 1, bias=False), nn.BatchNorm2d(bw), nn.GELU(),
            BinaryConv2d(bw, bw*2, 3, 2, 1, bias=False), nn.BatchNorm2d(bw*2), nn.GELU(),
            BinaryConv2d(bw*2, bw*4, 3, 2, 1, bias=False), nn.BatchNorm2d(bw*4), nn.GELU(),
            BinaryConv2d(bw*4, bw*4, 3, 1, 1, bias=False), nn.BatchNorm2d(bw*4), nn.GELU(),
            BinaryConv2d(bw*4, bw*4, 3, 1, 1, bias=False), nn.BatchNorm2d(bw*4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            BinaryLinear(bw*4, num_classes, bias=False),
        )
    def forward(self, x):
        gates, _ = self.gate(x)
        logits = torch.stack([e(x) for e in self.experts], dim=1)
        return (gates.unsqueeze(-1) * logits).sum(dim=1)


# Ablation registry
ABLATIONS = {
    "baseline": lambda **kw: BinaryMoE(**{**BEST_CONFIG, **kw}),
    "single_expert": lambda **kw: SingleExpertModel(**kw),
    "no_gate": lambda **kw: NoGateMoE(**kw),
    "random_gate": lambda **kw: RandomGateMoE(**kw),
    "all_experts": lambda **kw: AllExpertsMoE(**kw),
    "fp_experts": lambda **kw: FPExpertMoE(**kw),
    "all_binary": lambda **kw: AllBinaryExpertMoE(**kw),
}


def run_ablation(round_num: int, tag: str, ablation: str, overrides: dict = None) -> dict:
    set_seed(ITER_SEED)
    train_loader, test_loader = get_loaders()
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation}. Available: {list(ABLATIONS)}")
    model = ABLATIONS[ablation](**(overrides or {}))
    n_params = sum(p.numel() for p in model.parameters())
    # estimate binary bits (1 for binary modules, 32 for FP)
    n_bits = 0
    for name, p in model.named_parameters():
        bw = 32
        if any(t in name.lower() for t in ("binary", "lut_table")):
            bw = 1
        n_bits += p.numel() * bw

    optim = torch.optim.Adam(model.parameters(), lr=ITER_LR)
    trainer = Trainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        optimizer=optim, device="cpu", exp_name=f"abl.r{round_num}",
        seed=ITER_SEED, output_dir="./results", log_every=50, num_threads=2,
    )
    t0 = time.time()
    try:
        result = trainer.fit(epochs=ITER_EPOCHS)
        dt = time.time() - t0
        row = {
            "round": round_num, "tag": tag, "ablation": ablation,
            "params": n_params, "bits": n_bits,
            "best_top1": result.best.get("top1", 0.0),
            "final_top1": result.final.get("top1", 0.0),
            "final_loss": result.final.get("loss", 0.0),
            "time_s": round(dt, 1), "status": "ok", "error": "",
        }
    except Exception as e:
        dt = time.time() - t0
        row = {
            "round": round_num, "tag": tag, "ablation": ablation,
            "params": n_params, "bits": n_bits,
            "best_top1": 0.0, "final_top1": 0.0, "final_loss": 0.0,
            "time_s": round(dt, 1), "status": "fail", "error": str(e)[:200],
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
    p.add_argument("--ablation", type=str, default="baseline",
                   choices=list(ABLATIONS.keys()))
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--base-width", type=int, default=None)
    args = p.parse_args()

    overrides = {}
    if args.num_experts is not None: overrides["num_experts"] = args.num_experts
    if args.topk is not None: overrides["topk"] = args.topk
    if args.base_width is not None: overrides["base_width"] = args.base_width

    print(f"\n=== ABLATION ROUND {args.round}: {args.tag} ({args.ablation}) ===")
    row = run_ablation(args.round, args.tag, args.ablation, overrides or None)
    append_csv(row)
    print(f"\n=== RESULT: top1={row['best_top1']:.4f} loss={row['final_loss']:.4f} "
          f"params={row['params']:,} bits={row['bits']:,} time={row['time_s']}s ===")


if __name__ == "__main__":
    main()
