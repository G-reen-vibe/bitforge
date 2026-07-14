"""TinyBinaryExpert: a small binary CNN expert for MoE.

Architecture: 2-3 binary conv layers + global avg pool + binary linear.
Each expert is ~5-10K params so we can have many of them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitforge.models.baselines.layers import BinaryConv2d
from bitforge.models.baselines.layers import BinaryLinear


class TinyBinaryExpert(nn.Module):
    """A tiny binary CNN expert.

    Architecture:
        FP first conv (3x3) -> BN -> ReLU  (input is real-valued)
        Binary conv (3x3) -> BN -> ReLU
        Binary conv (3x3) -> BN -> ReLU
        Global avg pool -> Binary linear
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        base_width: int = 16,
    ):
        super().__init__()
        # FP first conv (per Bi-Real/XNOR-Net: keep first conv FP)
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        # Binary conv layers
        self.conv2 = BinaryConv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_width * 2)
        self.conv3 = BinaryConv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_width * 4)
        # Binary linear classifier
        self.fc = BinaryLinear(base_width * 4, num_classes, bias=False)
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
