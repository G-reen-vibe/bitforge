"""HDC encoder: project real-valued input to a dense binary hypervector.

Method (standard HDC vision encoding):
    1. For each spatial position, assign a fixed random base HV (position HV)
    2. For each channel, assign a fixed random channel HV
    3. For each pixel: bind = XOR(pixel_thresholded, position_hv, channel_hv)
    4. Bundle (majority) all pixel-HVs into a single D-dim hypervector

For efficiency on CPU, we use a simpler but equivalent scheme:
    - Random Gaussian projection R: (C*H*W) -> D
    - sign(R @ x.flatten()) gives the binary hypervector
    - R is fixed (no learning); ±1 after sign-binarization

This is the "random projection" HDC encoding from Frady et al. (2021) and
Kleyko et al. (2021). D is a hyperparameter (default 4096).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HDCEncoder(nn.Module):
    """Random-projection HDC encoder.

    Args:
        input_dim: flattened input size (C*H*W)
        hv_dim: hypervector dimension D (default 4096)
        seed: random seed for the projection (fixed across runs)
        binarize_input: if True, sign-binarize the input first (for BNN-style inputs)
    """

    def __init__(
        self,
        input_dim: int,
        hv_dim: int = 4096,
        seed: int = 42,
        binarize_input: bool = False,
        learnable: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        self.binarize_input = binarize_input
        # generate the random projection matrix with a fixed seed
        g = torch.Generator()
        g.manual_seed(seed)
        # use ±1 random matrix for memory efficiency (vs FP Gaussian)
        # Each entry: ±1 with equal probability
        R = torch.randint(0, 2, (input_dim, hv_dim), generator=g, dtype=torch.float32) * 2 - 1
        # scale by 1/sqrt(input_dim) for variance preservation
        R = R / (input_dim ** 0.5)
        if learnable:
            # Learnable latent projection; will be sign-binarized at inference
            self.R = nn.Parameter(R)
        else:
            self.register_buffer("R", R)
        self.learnable = learnable
        self._bit_width = 1  # output is binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> (N, D) binary hypervector."""
        N = x.size(0)
        x_flat = x.reshape(N, -1)
        if self.binarize_input:
            x_flat = torch.sign(x_flat)
        # project (FP matmul — this is the only FP op in the model)
        proj = x_flat @ self.R  # (N, D)
        # binarize to ±1; use STE if learnable
        if self.training:
            hv = torch.sign(proj)
            if self.learnable:
                hv = hv + (proj - proj.detach())
            # if any zero (unlikely), set to +1
            hv = torch.where(hv == 0, torch.ones_like(hv), hv)
        else:
            hv = torch.sign(proj)
            hv = torch.where(hv == 0, torch.ones_like(hv), hv)
        self._last_proj = proj  # stash for downstream use (e.g. readout)
        return hv

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hv_dim={self.hv_dim}, binarize_input={self.binarize_input}"
