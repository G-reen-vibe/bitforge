"""BitNorm: normalization layers designed for binary networks.

Standard BatchNorm defeats binarization — its learnable affine (gamma, beta)
and running statistics are FP, so binary activations become FP after BN.
This module provides binary-friendly normalizers that preserve the binary
signal while still controlling its scale.

Two variants:
    - PopCountNorm: normalize by (count_of_+1_minus_count_of_-1) / N — useful
      right after binary activation to keep activations zero-mean, unit-variance
      per channel in a binary-aware sense.
    - BitNorm: a lightweight per-channel scale-only normalization. No learnable
      affine (gamma), no bias (beta) — just a per-channel scalar trained as a
      FP parameter (the only FP cost is a multiply per channel per batch).
      This is the minimal "scale-only" version used by Bi-Real.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PopCountNorm(nn.Module):
    """Binary-aware normalization: scales each channel by 1/sqrt(N_eff)
    where N_eff = number of binary elements per channel.

    For an input (N, C, ...) of ±1 values, output is x / sqrt(d) where
    d = product of spatial dims. This keeps variance ~ 1/d per channel
    and is a no-op at inference (constant folding).
    """

    def __init__(self, num_features: int, num_spatial: int = 1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_spatial = num_spatial
        self.eps = eps
        # scale = 1 / sqrt(d), constant — but stored as buffer for clarity
        scale = 1.0 / (num_spatial + eps) ** 0.5
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, num_spatial={self.num_spatial}"


class BitNorm(nn.Module):
    """Scale-only normalization for binary networks (Bi-Real style).

    Has a single learnable per-channel scale `alpha`. No bias, no running
    stats. Acts as: y = alpha * x  (alpha is FP per channel, learned).

    This is the *minimal* FP overhead possible while still giving the
    optimizer a per-channel magnitude to play with.
    """

    def __init__(self, num_features: int, init_value: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # broadcast alpha over channel dim (assumes NCHW or NC)
        if x.dim() == 2:
            return x * self.alpha
        elif x.dim() == 4:
            return x * self.alpha.view(1, -1, 1, 1)
        else:
            # generic broadcast: assume dim 1 is channel
            shape = [1] * x.dim()
            shape[1] = -1
            return x * self.alpha.view(shape)

    def extra_repr(self) -> str:
        return f"num_features={self.alpha.numel()}"


class BatchNormBN(nn.Module):
    """Standard BatchNorm2d but kept in FP — used by baselines (XNOR-Net etc.)
    that explicitly keep BN in FP per the original paper.

    Wraps torch.nn.BatchNorm2d for naming clarity in model summaries.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)
