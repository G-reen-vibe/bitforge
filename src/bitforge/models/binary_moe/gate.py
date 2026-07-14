"""TopKGate: a small FP gating network that picks top-k experts per sample."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGate(nn.Module):
    """A small FP gating network.

    Takes the input image (or a downsampled version), produces a logit per
    expert, then applies top-k sparse routing (only top-k experts get
    non-zero weights).

    Args:
        in_channels, img_size: input image shape
        num_experts: number of experts to route between
        topk: number of experts to activate per sample (default 2)
        hidden_dim: gate hidden dim (small FP network)
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 28,
        num_experts: int = 4,
        topk: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        # Small FP conv->linear gate (kept FP for stable routing)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        # Add noise during training for load balancing (GShard-style)
        self.gate_noise = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gates, expert_indices).

        gates: (N, num_experts) sparse weights, sum to 1 over top-k
        expert_indices: (N, topk) indices of the top-k experts per sample
        """
        logits = self.gate(x)  # (N, num_experts)
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(logits) * self.gate_noise
            logits = logits + noise
        # top-k routing
        topk_logits, topk_idx = logits.topk(self.topk, dim=-1)  # (N, topk)
        # softmax over top-k only
        topk_weights = F.softmax(topk_logits, dim=-1)  # (N, topk)
        # scatter back to full expert dim
        gates = torch.zeros_like(logits).scatter_(-1, topk_idx, topk_weights)
        return gates, topk_idx
