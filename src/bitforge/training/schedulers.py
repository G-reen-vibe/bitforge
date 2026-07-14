"""Schedulers: cosine, step, and an annealing schedule for surrogate slopes."""
from __future__ import annotations

from typing import Optional

import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    LambdaLR,
    _LRScheduler,
)


def build_scheduler(optimizer, name: str = "cosine", **kwargs) -> Optional[_LRScheduler]:
    """Build a learning-rate scheduler by name.

    Supported:
        - cosine    : CosineAnnealingLR (T_max=epochs)
        - step      : StepLR (step_size, gamma)
        - multistep : MultiStepLR (milestones, gamma)
        - none      : returns None
    """
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        T_max = kwargs.get("T_max", kwargs.get("epochs", 100))
        eta_min = kwargs.get("eta_min", 0.0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    if name == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    if name == "multistep":
        milestones = kwargs.get("milestones", [60, 120])
        gamma = kwargs.get("gamma", 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name}")
