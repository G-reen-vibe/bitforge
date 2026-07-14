"""FP-ResNet: full-precision upper-bound baseline.

A small ResNet variant designed for low-resource CIFAR/MNIST experiments:
    - 3 stages of BasicBlock, each with 2 conv layers (3x3)
    - Channel widths: [16, 32, 64] * width_multiplier
    - First conv is 3x3 stride 1 (CIFAR) or 3x3 stride 1 with 1 channel (MNIST)
    - Global average pool -> linear classifier
    - Standard BN + ReLU

This is intentionally NOT a 50-layer ResNet — that's too heavy for our
2-core CPU. We need a model where the baselines and Hyper-LUT are on the
same compute budget.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2x 3x3 conv)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class FPResNet(nn.Module):
    """Full-precision ResNet for CIFAR/MNIST."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        img_size: int = 32,
        width_multiplier: float = 1.0,
        blocks_per_stage: int = 2,
        base_width: int = 16,
    ):
        super().__init__()
        widths = [int(base_width * w * width_multiplier) for w in (1, 2, 4)]
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = self._make_layer(widths[0], blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(widths[1], blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(widths[2], blocks_per_stage, stride=2)
        self.fc = nn.Linear(widths[2], num_classes)
        # for storage estimation
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m._bit_width = 32

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
