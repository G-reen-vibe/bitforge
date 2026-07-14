"""Tests for utility functions: metrics aggregation, config loading."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitforge.utils.metrics import (
    RunResult, compute_metrics, aggregate_runs, mean_pm_std
)
from bitforge.utils.config import Config


def test_compute_metrics_basic():
    logits = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]])
    labels = torch.tensor([0, 1, 1])
    m = compute_metrics(logits, labels, topk=(1,))
    assert 0.0 <= m["top1"] <= 1.0
    assert m["loss"] > 0


def test_compute_metrics_top1():
    # 2/3 correct
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1])  # 2nd is wrong
    m = compute_metrics(logits, labels, topk=(1,))
    assert abs(m["top1"] - 2/3) < 1e-6


def test_aggregate_runs():
    runs = [
        RunResult(seed=0, final={"top1": 0.80, "loss": 0.5}, best={"top1": 0.85, "loss": 0.4}),
        RunResult(seed=1, final={"top1": 0.82, "loss": 0.48}, best={"top1": 0.87, "loss": 0.38}),
        RunResult(seed=2, final={"top1": 0.78, "loss": 0.52}, best={"top1": 0.83, "loss": 0.42}),
    ]
    agg = aggregate_runs(runs)
    assert abs(agg["final.top1"]["mean"] - 0.80) < 1e-6
    assert agg["final.top1"]["n"] == 3
    assert agg["final.top1"]["std"] > 0
    s = mean_pm_std(agg, "final.top1")
    assert "±" in s


def test_config_dotpath():
    cfg = Config.from_dict({"model": {"name": "test", "k": 4}, "train": {"lr": 0.01}})
    assert cfg.get("model.name") == "test"
    assert cfg.get("model.k") == 4
    assert cfg.get("train.lr") == 0.01
    assert cfg.get("nonexistent", "default") == "default"
    cfg.set("model.k", 8)
    assert cfg.get("model.k") == 8
    # cfg.model should reflect cfg.set() mutations (live view)
    assert cfg.model["name"] == "test"
    assert cfg.model["k"] == 8
