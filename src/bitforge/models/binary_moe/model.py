"""BinaryMoE model: gate + N experts, top-k sparse routing."""
from __future__ import annotations

import torch
import torch.nn as nn

from bitforge.models.binary_moe.expert import TinyBinaryExpert
from bitforge.models.binary_moe.gate import TopKGate


class BinaryMoE(nn.Module):
    """Binary Mixture of Tiny CNN Experts.

    Args:
        in_channels, img_size, num_classes: input/output shape
        num_experts: how many binary experts
        topk: how many experts to activate per sample
        base_width: expert base channel width
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 28,
        num_classes: int = 10,
        num_experts: int = 4,
        topk: int = 2,
        base_width: int = 16,
        gate_noise: float = 1.0,
        shared_stem: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.shared_stem = shared_stem
        if shared_stem:
            # Shared FP conv stem before experts (experts then take this as input)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(),
            )
            expert_in = base_width
        else:
            expert_in = in_channels
        self.gate = TopKGate(expert_in, img_size, num_experts, topk)
        self.gate.gate_noise = gate_noise
        self.experts = nn.ModuleList([
            TinyBinaryExpert(expert_in, num_classes, base_width,
                             skip_first_fp=shared_stem)
            for _ in range(num_experts)
        ])
        for e in self.experts:
            e._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> logits: (N, num_classes)"""
        if self.shared_stem:
            x = self.stem(x)
        gates, expert_idx = self.gate(x)  # (N, E), (N, topk)
        expert_logits = torch.stack([e(x) for e in self.experts], dim=1)  # (N, E, C)
        out = (gates.unsqueeze(-1) * expert_logits).sum(dim=1)  # (N, C)
        # stash gates for load-balance loss
        self._last_gates = gates
        return out

    def load_balance_loss(self) -> torch.Tensor:
        """Standard MoE load-balancing loss.

        L = E * sum_i (f_i * P_i) where:
          f_i = fraction of tokens routed to expert i
          P_i = mean gate prob for expert i
        Minimizing this encourages uniform usage.
        """
        if not hasattr(self, "_last_gates") or self._last_gates is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        gates = self._last_gates  # (N, E)
        N, E = gates.shape
        # f_i = fraction of tokens where expert i is in topk
        # (gates > 0 means expert was selected)
        f = (gates > 0).float().mean(dim=0)  # (E,)
        # P_i = mean gate prob for expert i
        P = gates.mean(dim=0)  # (E,)
        return E * (f * P).sum()
