"""ReActNet (Liu et al., 2020): ReAct-Net uses learnable shifting (RSign) and
learnable ReLU shifting (RPreF) to make binarization more flexible.

Key contributions reproduced here:
    - RSign: sign(x - beta) with learnable per-channel beta (replaces plain sign)
    - RPreF: ReLU with learnable per-channel shift, applied before binary conv
    - Uses a binary ResNet-18-like backbone with these learnable shifts

For our small-CPU setting we use a ResNet-9-ish backbone (3 stages x 2 blocks)
to keep compute feasible. The RSign/RPreF mechanism is the defining feature.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitforge.models.baselines.layers import BinaryConv2d
from bitforge.training.bitnorm import BatchNormBN


class RSign(nn.Module):
    """ReActNet's learnable sign: sign(x - beta) with per-channel beta."""

    def __init__(self, num_features: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # broadcast beta
        if x.dim() == 4:
            beta = self.beta.view(1, -1, 1, 1)
        else:
            beta = self.beta.view(1, -1)
        if self.training:
            # STE on (x - beta)
            x_shift = x - beta
            forward_val = torch.sign(x_shift)
            return forward_val + (x_shift - x_shift.detach())
        return torch.sign(x - beta)


class RPreF(nn.Module):
    """ReActNet's learnable ReLU with shift: ReLU(x - beta) then back: + beta.

    Applied before binary conv to keep activations in a useful range.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            beta = self.beta.view(1, -1, 1, 1)
        else:
            beta = self.beta.view(1, -1)
        return F.relu(x - beta) + beta


class ReActBlock(nn.Module):
    """ReActNet BasicBlock: RPreF -> RSign -> BinConv -> BN -> RPreF -> RSign -> BinConv -> BN."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.pref0 = RPreF(in_planes)
        self.rsign0 = RSign(in_planes)
        self.conv1 = BinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, scale=False)
        self.bn1 = BatchNormBN(planes)
        self.pref1 = RPreF(planes)
        self.rsign1 = RSign(planes)
        self.conv2 = BinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, scale=False)
        self.bn2 = BatchNormBN(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.rsign0(self.pref0(x))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.rsign1(self.pref1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        return out


class ReActNet(nn.Module):
    """ReActNet for CIFAR/MNIST (small backbone for CPU)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        img_size: int = 32,
        width_multiplier: float = 1.0,
        blocks_per_stage: int = 2,
        base_width: int = 32,
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
        for m in [self.conv1, self.fc]:
            m._bit_width = 32

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(ReActBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
