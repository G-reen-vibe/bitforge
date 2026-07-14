"""Bundling (majority vote) operations for hypervectors."""
from __future__ import annotations

import torch


def majority_bundle(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Bundle a tensor of binary hypervectors via majority vote.

    For input (..., D) of ±1 values, the bundle is sign(sum(x, dim)).
    Ties (sum=0) are broken to +1 by convention.

    Args:
        x: input tensor of ±1 values
        dim: dimension to reduce (default: last)
    """
    s = x.sum(dim=dim)
    out = torch.sign(s)
    # tie-break to +1
    out = torch.where(out == 0, torch.ones_like(out), out)
    return out


def soft_majority_bundle(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Soft (tanh-based) majority: tanh(sum(x, dim)).

    Differentiable version of majority_bundle. Output is in (-1, 1).
    """
    return torch.tanh(x.sum(dim=dim))
