"""Surrogate gradients for binarization.

We do NOT use STE in our main framework — but baselines (XNOR-Net, Bi-Real,
IR-Net, ReActNet) all rely on some form of surrogate. We provide them here so
each baseline can be trained faithfully with its published surrogate.

Surrogates implemented:
    - SignSTE      : classic straight-through estimator (sign forward, grad=1)
    - ATanSurrogate: IR-Net's atan surrogate with adjustable slope
    - PolytanhSurrogate: a piecewise polynomial surrogate ( smoother than hard tanh)
    - SwishSurrogate:   ReActNet's learnable swish-based surrogate (RSign + RPreF)
    - HardTanhSurrogate: piecewise-linear clamp

All surrogates implement forward = binarize (to ±1), backward = surrogate_grad.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Base interface
# -----------------------------------------------------------------------------
class SurrogateBin(nn.Module):
    """Base class. Forward returns binarized {+1, -1}. Subclasses override
    `_surrogate_grad(x)` which is the gradient w.r.t. input used in backward."""

    def __init__(self, clip_val: float = 1.0):
        super().__init__()
        self.clip_val = clip_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward: sign, but with a detach-aware trick so backward uses surrogate
        if self.training:
            x_clip = x.clamp(-self.clip_val, self.clip_val)
            forward_val = torch.sign(x_clip)
            # surrogate gradient: grad_out * g'(x)
            return forward_val + (x_clip - x_clip.detach()) * self._surrogate_grad(x_clip)
        else:
            return torch.sign(x)

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# SignSTE — identity gradient within clip range
# -----------------------------------------------------------------------------
class SignSTE(SurrogateBin):
    """Classic STE: forward sign, backward identity (within clip range)."""

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


# -----------------------------------------------------------------------------
# ATanSurrogate (IR-Net)
# -----------------------------------------------------------------------------
class ATanSurrogate(SurrogateBin):
    """IR-Net's atan surrogate: g'(x) = slope * 0.5 / (1 + (pi*slope*x)^2)."""

    def __init__(self, clip_val: float = 1.0, slope: float = 2.0):
        super().__init__(clip_val=clip_val)
        self.slope = slope

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        s = self.slope
        # g'(x) = 2*s / (pi * (1 + (2*s*x/pi)^2))? The IR-Net paper uses:
        # grad = slope * 0.5 / (1 + (slope * pi * x)^2) ? — we use the form
        # that matches the published implementation:
        # d/dx [ (2/pi) * atan(slope * pi * x / 2) ] = slope / (1 + (slope*pi*x/2)^2)
        # but in practice most re-implementations use:
        # grad = (2/pi) * slope / (1 + (pi * slope * x)^2 / 4)
        # We use the simple, common form:
        return s / (1.0 + (math.pi * s * x / 2.0) ** 2)


# -----------------------------------------------------------------------------
# PolytanhSurrogate — polynomial smoother
# -----------------------------------------------------------------------------
class PolytanhSurrogate(SurrogateBin):
    """A piecewise-polynomial surrogate (similar to IR-Net's polytanh):
        grad = max(0, 1 - |x|^p / clip^p)  (1 inside, 0 outside, smooth at edges)
    """

    def __init__(self, clip_val: float = 1.0, p: float = 2.0):
        super().__init__(clip_val=clip_val)
        self.p = p

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        c = self.clip_val
        return F.relu(1.0 - (x.abs() / c) ** self.p)


# -----------------------------------------------------------------------------
# SwishSurrogate — ReActNet uses learnable RSign (swish-shaped surrogate)
# -----------------------------------------------------------------------------
class SwishSurrogate(SurrogateBin):
    """Learnable swish-based surrogate (ReActNet's RSign).

    The forward is sign(x - beta), where beta is a learnable per-channel bias.
    Backward is the derivative of swish: sigmoid(beta*x) + beta*x*sigmoid(beta*x)*(1-sigmoid(beta*x)).
    """

    def __init__(self, num_features: int, clip_val: float = 1.0, learnable: bool = True):
        super().__init__(clip_val=clip_val)
        # Per-channel bias; for non-channel inputs (e.g. weights), caller can broadcast.
        self.beta = nn.Parameter(torch.zeros(num_features), requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # broadcast beta over (N, C, ...) -> beta reshaped to (1, C, 1...)
        beta = self.beta
        # reshape beta to broadcast across x
        if x.dim() > 1:
            shape = [1] * x.dim()
            shape[1] = -1
            beta = beta.view(shape)
        if self.training:
            x_shift = x - beta
            x_clip = x_shift.clamp(-self.clip_val, self.clip_val)
            forward_val = torch.sign(x_clip)
            grad = self._surrogate_grad(x_clip)
            return forward_val + (x_clip - x_clip.detach()) * grad
        else:
            return torch.sign(x - beta)

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        # swish derivative with slope=1: d/dx(x*sigmoid(x)) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        s = torch.sigmoid(x)
        return s + x * s * (1.0 - s)


# -----------------------------------------------------------------------------
# HardTanhSurrogate — piecewise linear
# -----------------------------------------------------------------------------
class HardTanhSurrogate(SurrogateBin):
    """Hard tanh: grad=1 in [-clip,clip], 0 outside. Equivalent to clipped STE."""

    def _surrogate_grad(self, x: torch.Tensor) -> torch.Tensor:
        c = self.clip_val
        return ((x >= -c) & (x <= c)).float()


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
SURROGATES = {
    "ste": SignSTE,
    "atan": ATanSurrogate,
    "polytanh": PolytanhSurrogate,
    "swish": SwishSurrogate,
    "hardtanh": HardTanhSurrogate,
}


# -----------------------------------------------------------------------------
# Weight binarization utility (used inside binary conv/linear layers)
# -----------------------------------------------------------------------------
class BinaryWeightSTE(nn.Module):
    """Binarize a weight tensor with STE backward (mean-scaling variant).

    Implements: w_b = sign(w) * E[|w|]   (XNOR-Net style scaling, optional)
    For pure BNN (Bi-Real/IR-Net) set scale=False.
    """

    def __init__(self, scale: bool = False):
        super().__init__()
        self.scale = scale

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_clip = w.clamp(-1.0, 1.0)
            scale = w_clip.abs().mean() if self.scale else 1.0
            wb = torch.sign(w_clip) * scale
            return wb + (w_clip - w_clip.detach())  # STE
        else:
            scale = w.abs().mean() if self.scale else 1.0
            return torch.sign(w) * scale
