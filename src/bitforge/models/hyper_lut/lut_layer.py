"""Differentiable LUT layer: a k-input lookup table trained via Gumbel-softmax.

A single LUT has 2^k entries (a truth table). Each entry is a learned
logit; the output is a weighted combination (Gumbel-softmax) of the binary
values that the entry can take. As temperature -> 0, the LUT becomes
deterministic (one entry selected, output = ±1).

To apply a LUT to a D-dim hypervector:
    - Take k coordinates at a time (with optional stride / overlap)
    - For each k-tuple, compute the integer index (binary -> int)
    - Look up the (soft) value from the table

For a D=4096 HV with k=4 and stride=4, we get 1024 outputs per LUT.
We use multiple LUT "channels" (like conv filters) to produce a richer
output, then bundle (majority) them back to a D-dim HV for the next block.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableLUT(nn.Module):
    """A bank of differentiable LUTs over k binary inputs.

    Args:
        k: number of inputs per LUT (table size = 2^k)
        num_luts: number of parallel LUTs (like out_channels in conv)
        init_temp: initial Gumbel-softmax temperature (high = soft)
        min_temp: minimum temperature at the end of annealing
        hard: if True, use straight-through Gumbel (hard sample + STE for grad)
    """

    def __init__(
        self,
        k: int = 4,
        num_luts: int = 64,
        init_temp: float = 1.0,
        min_temp: float = 0.3,
        hard: bool = False,
    ):
        super().__init__()
        assert 1 <= k <= 8, f"k={k} too large (2^k entries)"
        self.k = k
        self.num_luts = num_luts
        self.table_size = 2 ** k
        self.hard = hard
        # logits for each entry of each LUT: shape (num_luts, 2^k)
        # init to N(0, 1) so initial soft outputs are not too close to 0
        self.logits = nn.Parameter(torch.randn(num_luts, self.table_size))
        # temperature is a buffer (managed externally by the trainer)
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.min_temp = float(min_temp)
        self._bit_width = 1

    def set_temperature(self, temp: float) -> None:
        self.temperature.fill_(max(temp, self.min_temp))

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        """Apply LUT bank to k-bit inputs.

        Args:
            x_k: (..., k) tensor of ±1 binary values (k bits per LUT input)
        Returns:
            (..., num_luts) tensor of soft binary outputs (in [-1, 1] when
            temperature is low; approximately ±1 when temperature -> 0)
        """
        # Convert ±1 binary to {0,1} bits, then to integer index in [0, 2^k)
        bits = (x_k > 0).long()  # (..., k)
        # Compute integer index: bit i contributes 2^i (MSB-first or LSB-first?)
        # We use MSB-first to match standard truth-table indexing.
        *leading, k_dim = bits.shape
        # weights for index computation: 2^(k-1), 2^(k-2), ..., 1
        weights = (1 << torch.arange(k_dim - 1, -1, -1, device=bits.device)).long()
        idx = (bits * weights).sum(dim=-1)  # (...,)
        # Gather logits at this index: shape (..., num_luts)
        # logits: (num_luts, 2^k) -> transpose -> (2^k, num_luts)
        gathered = self.logits.t()[idx]  # (..., num_luts)
        # Use tanh + STE for binarization (simpler than 2-way Gumbel, more
        # stable gradients). The temperature scales the logit before tanh.
        if self.training:
            scaled = gathered / self.temperature
            out = torch.tanh(scaled)
            # STE: forward is tanh, backward is identity (clipped)
            # This is the standard surrogate for sign()
            out_bin = torch.sign(out)
            return out_bin + (out - out.detach())
        else:
            # at eval: hard binarize
            return torch.sign(gathered)


class MultiKLUTBlock(nn.Module):
    """A block with PARALLEL LUTs at multiple k values (k=2, 3, 4) for richer
    feature extraction. Outputs from each k-head are concatenated, then
    projected back to hv_dim.

    This gives each block multiple "receptive fields" over the input HV.
    """

    def __init__(
        self,
        hv_dim: int,
        ks: tuple = (2, 3, 4),
        num_luts_per_k: int = 16,
        permute: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.hv_dim = hv_dim
        self.ks = ks
        self.luts = nn.ModuleList([
            DifferentiableLUT(k=k, num_luts=num_luts_per_k, hard=False)
            for k in ks
        ])
        # total output dim: sum over k of (n_groups_k * num_luts_per_k)
        self.n_groups_per_k = [(hv_dim - k) // k + 1 for k in ks]
        self.out_dim = sum(n * num_luts_per_k for n in self.n_groups_per_k)
        self.proj = nn.Linear(self.out_dim, hv_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / (self.out_dim ** 0.5))
        self.permute = permute
        if permute:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(hv_dim, generator=g)
            self.register_buffer("perm", perm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        outs = []
        for lut, k, n_groups in zip(self.luts, self.ks, self.n_groups_per_k):
            groups = x.unfold(1, k, k)  # (N, n_groups, k)
            out = lut(groups)  # (N, n_groups, num_luts)
            outs.append(out.reshape(N, -1))
        out = torch.cat(outs, dim=-1)  # (N, out_dim)
        out = self.proj(out)  # (N, hv_dim)
        if self.training:
            out_bin = torch.sign(out)
            out = out_bin + (out - out.detach())
        else:
            out = torch.sign(out)
        if self.permute:
            out = out[:, self.perm]
        return out

    def set_temperature(self, temp: float) -> None:
        for lut in self.luts:
            lut.set_temperature(temp)


class LUTBlock(nn.Module):
    """A block: take D-dim HV, apply LUT-bank, bundle back to D-dim HV.

    Args:
        hv_dim: D, dimension of input/output hypervector
        k: LUT input size
        num_luts: number of parallel LUTs
        stride: stride between k-tuples (default = k, no overlap)
        permute: if True, apply a fixed permutation to HV between blocks
    """

    def __init__(
        self,
        hv_dim: int,
        k: int = 4,
        num_luts: int = 64,
        stride: int = None,
        permute: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.hv_dim = hv_dim
        self.k = k
        self.num_luts = num_luts
        self.stride = stride or k
        assert (hv_dim - k) % self.stride == 0 or hv_dim >= k, "HV dim must accommodate stride"
        self.n_groups = (hv_dim - k) // self.stride + 1
        # output dim per block: n_groups * num_luts
        self.out_dim = self.n_groups * num_luts
        self.lut = DifferentiableLUT(k=k, num_luts=num_luts, hard=False)
        # Learnable per-coordinate bias (shift) and scale applied BEFORE LUT.
        # Gives the LUT inputs a learnable offset so the binarization threshold
        # can shift per coordinate. Similar to ReActNet's RSign.
        self.pre_bias = nn.Parameter(torch.zeros(hv_dim))
        # FP projection back to hv_dim (kept FP for stability during Gumbel training,
        # will be binarized at inference via sign() — see forward()).
        # Small init (std = 1/sqrt(out_dim)) keeps initial outputs in unit range.
        self.proj = nn.Linear(self.out_dim, hv_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / (self.out_dim ** 0.5))
        # fixed permutation (HDC orthogonal reorder)
        self.permute = permute
        if permute:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(hv_dim, generator=g)
            self.register_buffer("perm", perm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) -> (N, D)"""
        N = x.size(0)
        # apply learnable shift (RSign-style) — shifts the effective binarization threshold
        x_shifted = x - self.pre_bias
        # re-binarize with STE
        if self.training:
            x_bin = torch.sign(x_shifted)
            x_shifted = x_bin + (x_shifted - x_shifted.detach())
        else:
            x_shifted = torch.sign(x_shifted)
        # extract sliding groups of k bits
        groups = x_shifted.unfold(1, self.k, self.stride)  # (N, n_groups, k)
        # apply LUT to each group -> (N, n_groups, num_luts)
        out = self.lut(groups)
        # flatten and project back to hv_dim
        out = out.reshape(N, -1)
        out = self.proj(out)
        if self.training:
            out_bin = torch.sign(out)
            out = out_bin + (out - out.detach())
        else:
            out = torch.sign(out)
        if self.permute:
            out = out[:, self.perm]
        return out

    def set_temperature(self, temp: float) -> None:
        self.lut.set_temperature(temp)
