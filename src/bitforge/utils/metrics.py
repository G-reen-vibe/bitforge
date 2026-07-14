"""Metrics: top-k accuracy, per-class accuracy, and aggregation across seeds."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, topk: Sequence[int] = (1,)) -> Dict[str, float]:
    """Compute top-k accuracy from logits and integer labels.

    Args:
        logits: (N, C) float tensor
        labels: (N,) long tensor
        topk: tuple of k values (e.g., (1,) or (1, 5))
    Returns:
        dict like {"top1": 0.85, "top5": 0.99}
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    max_k = max(topk)
    batch_size = labels.size(0)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    out = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = (correct_k / batch_size).item()
        out[f"top{k}"] = acc
    # also loss if labels are class indices (useful for tracking)
    out["loss"] = F.cross_entropy(logits, labels).item()
    return out


@dataclass
class RunResult:
    """Single-run result: final-epoch + best-epoch metrics."""
    seed: int
    final: Dict[str, float] = field(default_factory=dict)
    best: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    train_time_s: float = 0.0
    n_params: int = 0
    n_bits: int = 0  # estimated bits of storage


def aggregate_runs(runs: List[RunResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of RunResult (with same metric keys) into mean ± std.

    Returns nested dict: {metric_name: {"mean": x, "std": y, "min": z, "max": w, "n": n}}
    Aggregates over both `final` and `best` (prefix accordingly).
    """
    out: Dict[str, Dict[str, float]] = {}
    for prefix in ("final", "best"):
        # collect keys
        all_keys: set = set()
        for r in runs:
            src = getattr(r, prefix)
            all_keys.update(src.keys())
        for k in all_keys:
            vals = [float(getattr(r, prefix)[k]) for r in runs if k in getattr(r, prefix)]
            if not vals:
                continue
            arr = np.array(vals)
            out[f"{prefix}.{k}"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": len(arr),
            }
    # timing & params
    times = np.array([r.train_time_s for r in runs])
    out["train_time_s"] = {
        "mean": float(times.mean()),
        "std": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "min": float(times.min()),
        "max": float(times.max()),
        "n": len(times),
    }
    return out


def mean_pm_std(agg: Dict[str, Dict[str, float]], key: str, fmt: str = "{:.4f} ± {:.4f}") -> str:
    """Format mean ± std from aggregated dict (or 'N/A' if missing)."""
    if key not in agg:
        return "N/A"
    a = agg[key]
    return fmt.format(a["mean"], a["std"])
