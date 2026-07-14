"""Hyper-LUT Net: full model assembling encoder + LUT blocks + readout.

This is the *skeleton* — design iteration (k, num_luts, num_blocks,
temperature schedule, projection binarization) will happen in a later
milestone after baselines are validated.

Defaults are chosen to be CPU-friendly:
    - D (hv_dim): 2048 (not 8192 — memory)
    - k=4 (16-entry LUTs)
    - num_luts=32 per block
    - 4 LUT blocks
    - Binary linear readout
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitforge.models.hyper_lut.encoder import HDCEncoder
from bitforge.models.hyper_lut.lut_layer import LUTBlock
from bitforge.models.baselines.layers import BinaryLinear
from bitforge.training.bitnorm import BitNorm


class HyperLUTNet(nn.Module):
    """The Hyper-LUT Net model.

    Pipeline:
        x (N, C, H, W)
        -> HDC encoder -> binary HV (N, D)
        -> N x [LUTBlock + BitNorm] -> binary HV (N, D)
        -> BinaryLinear -> logits (N, num_classes)

    Args:
        in_channels, img_size: input image shape (used to compute input_dim)
        num_classes: classification head
        hv_dim: D, hypervector dimension (default 2048)
        k: LUT input bits (default 4)
        num_luts: number of LUTs per block (default 32)
        num_blocks: depth (default 4)
        encoder_seed: seed for the random projection (fixed, not learned)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        img_size: int = 32,
        hv_dim: int = 2048,
        k: int = 4,
        num_luts: int = 32,
        num_blocks: int = 4,
        encoder_seed: int = 42,
        binarize_input: bool = False,
    ):
        super().__init__()
        self.hv_dim = hv_dim
        self.num_blocks = num_blocks
        input_dim = in_channels * img_size * img_size
        self.encoder = HDCEncoder(
            input_dim=input_dim,
            hv_dim=hv_dim,
            seed=encoder_seed,
            binarize_input=binarize_input,
        )
        self.blocks = nn.ModuleList([
            LUTBlock(hv_dim=hv_dim, k=k, num_luts=num_luts, permute=True, seed=encoder_seed + i)
            for i in range(num_blocks)
        ])
        self.norms = nn.ModuleList([BitNorm(hv_dim) for _ in range(num_blocks)])
        # Binary linear readout
        self.fc = BinaryLinear(hv_dim, num_classes, bias=False, scale=False)
        # FP readout is also supported via a flag, but default is binary
        for m in [self.fc]:
            m._bit_width = 1

    def set_temperature(self, temp: float) -> None:
        """Anneal Gumbel temperature across all LUT blocks."""
        for blk in self.blocks:
            blk.set_temperature(temp)

    def get_temperature_schedule(self, epoch: int, total_epochs: int) -> float:
        """Linear anneal from init_temp (2.0) to min_temp (0.1) over training."""
        init_t, min_t = 2.0, 0.1
        if total_epochs <= 1:
            return min_t
        progress = min(epoch / max(1, total_epochs - 1), 1.0)
        return init_t + (min_t - init_t) * progress

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hv = self.encoder(x)  # (N, D) binary
        for blk, norm in zip(self.blocks, self.norms):
            hv_new = blk(hv)
            hv_new = norm(hv_new)
            # residual on the HV (add then re-binarize)
            hv = torch.sign(hv + hv_new)
        return self.fc(hv)

    def extra_repr(self) -> str:
        return f"hv_dim={self.hv_dim}, num_blocks={self.num_blocks}"
