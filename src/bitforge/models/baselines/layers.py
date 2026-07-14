"""Binary layer primitives: BinaryConv2d, BinaryLinear, HardBinaryConv2d, etc.

These layers keep a latent FP weight `w` and binarize it on each forward
during training (with STE-like backward). At eval time, weights are pure ±1
so XNOR-popcount kernels could be used (here we still call F.conv2d for
portability; the actual XNOR-popcount is documented in the inference path).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitforge.training.surrogates import BinaryWeightSTE


class BinaryConv2d(nn.Module):
    """A 2D conv whose weights are binarized to ±1 on each forward.

    Args:
        in_channels, out_channels, kernel_size, stride, padding, bias
        scale: if True, applies XNOR-Net-style per-output-channel scaling
               using mean(|w|) — only for XNOR-Net baseline.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        scale: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        # latent FP weight
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self._bit_width = 1  # for storage estimation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_b = BinaryWeightSTE(scale=self.scale)(self.weight)
        else:
            w_b = torch.sign(self.weight)
            if self.scale:
                w_b = w_b * self.weight.abs().mean()
        return F.conv2d(x, w_b, bias=self.bias, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    """Linear layer with binarized weights (±1). Inputs are assumed already binarized."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, scale: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_b = BinaryWeightSTE(scale=self.scale)(self.weight)
        else:
            w_b = torch.sign(self.weight)
            if self.scale:
                w_b = w_b * self.weight.abs().mean()
        return F.linear(x, w_b, bias=self.bias)


class HardBinaryConv2d(nn.Module):
    """Hard-binarized conv: weights binarized at eval AND train (no STE).

    Used in ablations / as a non-trainable reference. NOT used in main
    baselines — included for completeness.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self._bit_width = 1

    def forward(self, x):
        w = torch.sign(self.weight)
        return F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)
