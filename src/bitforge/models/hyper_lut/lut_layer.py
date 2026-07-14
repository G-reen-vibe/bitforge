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
        # Apply Gumbel-softmax on a binary {0,1} choice (interpret logits as
        # the logit for "1"; the logit for "-1" is -logits)
        # The Gumbel-softmax trick:
        #   softmax([+logit, -logit] / T) gives Bernoulli-ish probability
        # We implement the 2-way Gumbel:
        if self.training:
            # 2-way Gumbel-softmax on [+logit, -logit]
            two_logits = torch.stack([gathered, -gathered], dim=-1) / self.temperature  # (..., num_luts, 2)
            soft = F.gumbel_softmax(two_logits, tau=self.temperature, hard=self.hard, dim=-1)
            # soft[..., 0] is prob of +1, soft[..., 1] is prob of -1
            # binary output: +1 if [0] chosen, -1 if [1] chosen
            out = soft[..., 0] - soft[..., 1]  # (..., num_luts)
            return out
        else:
            # at eval: hard binarize
            return torch.sign(gathered)


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
        # extract sliding groups of k bits
        # x: (N, D) -> (N, n_groups, k)
        groups = x.unfold(1, self.k, self.stride)  # (N, n_groups, k)
        # apply LUT to each group -> (N, n_groups, num_luts)
        out = self.lut(groups)
        # flatten and project back to hv_dim
        out = out.reshape(N, -1)  # (N, n_groups * num_luts)
        out = self.proj(out)  # (N, hv_dim)
        # binarize back to ±1 (STE for proj weights)
        if self.training:
            out_bin = torch.sign(out)
            out = out_bin + (out - out.detach())
        else:
            out = torch.sign(out)
        # apply permutation (re-order coordinates)
        if self.permute:
            out = out[:, self.perm]
        return out

    def set_temperature(self, temp: float) -> None:
        self.lut.set_temperature(temp)
