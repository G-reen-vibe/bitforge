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
        spatial: bool = False,
        img_size: int = None,
        in_channels: int = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        self.binarize_input = binarize_input
        self.spatial = spatial
        if spatial:
            assert img_size is not None and in_channels is not None
            self.img_size = img_size
            self.in_channels = in_channels
            # Position HVs: one random ±1 HV per spatial position (H*W of them)
            g = torch.Generator()
            g.manual_seed(seed)
            self.position_hvs = torch.randint(0, 2, (img_size * img_size, hv_dim),
                                               generator=g, dtype=torch.float32) * 2 - 1
            # Channel HVs: one random ±1 HV per channel
            self.channel_hvs = torch.randint(0, 2, (in_channels, hv_dim),
                                              generator=g, dtype=torch.float32) * 2 - 1
            self.register_buffer("position_hvs_buf", self.position_hvs)
            self.register_buffer("channel_hvs_buf", self.channel_hvs)
            # Learnable per-pixel scale (latent FP, binarized at inference)
            # This is what gets trained — the binding weight per pixel
            self.pixel_logits = nn.Parameter(torch.zeros(in_channels * img_size * img_size))
        else:
            # generate the random projection matrix with a fixed seed
            g = torch.Generator()
            g.manual_seed(seed)
            R = torch.randint(0, 2, (input_dim, hv_dim), generator=g, dtype=torch.float32) * 2 - 1
            R = R / (input_dim ** 0.5)
            if learnable:
                self.R = nn.Parameter(R)
            else:
                self.register_buffer("R", R)
        self.learnable = learnable
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> (N, D) binary hypervector."""
        N = x.size(0)
        if self.spatial:
            # HDC spatial encoding: for each pixel (c, h, w),
            #   bind = XOR(channel_hv[c], position_hv[h*W+w])
            #   scale by pixel intensity (after tanh)
            # bundle = sum over all pixels
            C, H, W = self.in_channels, self.img_size, self.img_size
            x_flat = x.reshape(N, C * H * W)  # (N, C*H*W)
            # pixel intensity -> tanh-scaled contribution
            pixel_scale = torch.tanh(x_flat)  # (N, C*H*W)
            # bind: for each pixel, channel_hv XOR position_hv = channel_hv * position_hv (for ±1)
            # binding_hvs: (C*H*W, D) = (channel_hvs[:,None] * position_hvs[None,:]) reshaped
            # Precompute this once
            if not hasattr(self, "_binding_cache"):
                # binding_hvs[c*H*W + h*W + w] = channel_hvs[c] * position_hvs[h*W+w]
                ch = self.channel_hvs_buf.unsqueeze(1)  # (C, 1, D)
                pos = self.position_hvs_buf.unsqueeze(0)  # (1, HW, D)
                binding = (ch * pos).reshape(C * H * W, -1)  # (C*H*W, D)
                self._binding_cache = binding
            binding = self._binding_cache  # (C*H*W, D)
            # bundle: sum over pixels, weighted by pixel_scale and pixel_logits
            # (N, C*H*W) * (C*H*W, D) -> (N, D)
            # apply learnable per-pixel logit (tanh -> ±1 scale)
            pixel_weight = torch.tanh(self.pixel_logits)  # (C*H*W,)
            weighted = pixel_scale * pixel_weight.unsqueeze(0)  # (N, C*H*W)
            proj = weighted @ binding  # (N, D)
        else:
            x_flat = x.reshape(N, -1)
            if self.binarize_input:
                x_flat = torch.sign(x_flat)
            proj = x_flat @ self.R
        # binarize to ±1; use STE if learnable
        if self.training:
            hv = torch.sign(proj)
            if self.learnable or self.spatial:
                hv = hv + (proj - proj.detach())
            hv = torch.where(hv == 0, torch.ones_like(hv), hv)
        else:
            hv = torch.sign(proj)
            hv = torch.where(hv == 0, torch.ones_like(hv), hv)
        self._last_proj = proj
        return hv

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hv_dim={self.hv_dim}, binarize_input={self.binarize_input}"
