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
        learnable_encoder: bool = True,
        multi_k: bool = False,
        ks: tuple = (2, 3, 4),
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
            learnable=learnable_encoder,
        )
        from bitforge.models.hyper_lut.lut_layer import MultiKLUTBlock
        if multi_k:
            self.blocks = nn.ModuleList([
                MultiKLUTBlock(hv_dim=hv_dim, ks=ks, num_luts_per_k=num_luts // len(ks),
                               permute=True, seed=encoder_seed + i)
                for i in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                LUTBlock(hv_dim=hv_dim, k=k, num_luts=num_luts, permute=True, seed=encoder_seed + i)
                for i in range(num_blocks)
            ])
        self.multi_k = multi_k
        self.norms = nn.ModuleList([BitNorm(hv_dim) for _ in range(num_blocks)])
        # Learnable residual mixing: hv = tanh(alpha * hv + (1-alpha) * hv_new)
        # alpha initialized to 0.5 (equal mixing), can drift toward 0 (full replace)
        # or 1 (ignore block). Sigmoid keeps it in (0, 1).
        self.mix_alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
            for _ in range(num_blocks)
        ])
        # Binary linear readout
        self.fc = BinaryLinear(hv_dim, num_classes, bias=False, scale=False)
        # Fixed logit scale: 1/sqrt(D) keeps logits in unit-variance range
        # when hv is ±1 and fc weights are ±1 (dot product has std ~sqrt(D)).
        self.register_buffer("logit_scale", torch.tensor(1.0 / (hv_dim ** 0.5)))
        for m in [self.fc]:
            m._bit_width = 1

    def set_temperature(self, temp: float) -> None:
        """Anneal Gumbel temperature across all LUT blocks."""
        for blk in self.blocks:
            blk.set_temperature(temp)

    def get_temperature_schedule(self, epoch: int, total_epochs: int) -> float:
        """Linear anneal from init_temp (1.0) to min_temp (0.3) over training."""
        init_t, min_t = 1.0, 0.3
        if total_epochs <= 1:
            return min_t
        progress = min(epoch / max(1, total_epochs - 1), 1.0)
        return init_t + (min_t - init_t) * progress

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hv = self.encoder(x)  # (N, D) binary ±1
        for i, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
            hv_new = norm(blk(hv))
            alpha = torch.sigmoid(self.mix_alpha[i])
            # mix in FP space then tanh-bounded
            hv = torch.tanh(alpha * hv + (1.0 - alpha) * hv_new)
        # final binarization for the readout (STE)
        if self.training:
            hv_bin = torch.sign(hv)
            hv = hv_bin + (hv - hv.detach())
        else:
            hv = torch.sign(hv)
        return self.logit_scale * self.fc(hv)

    def regularization_loss(self) -> torch.Tensor:
        """Penalty pushing FP latent weights toward ±1: sum(1 - w^2) for |w|<1.

        Disabled by default (returns 0). Enable by setting the coefficient > 0
        in the trainer.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def extra_repr(self) -> str:
        return f"hv_dim={self.hv_dim}, num_blocks={self.num_blocks}"
