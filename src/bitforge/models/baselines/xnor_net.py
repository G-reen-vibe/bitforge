"""XNOR-Net (Rastegari et al., 2016): binary weights AND binary activations,
with per-channel mean-|w| scaling on weights and per-image scaling on activations.

For fair comparison on CPU, we implement the binary conv with mean-|w| scale
on weights. The activation-scaling (per-image, k-threshold) is approximated
via a simple ABS-mean scale — the exact original formulation is computationally
expensive on CPU.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitforge.models.baselines.layers import BinaryConv2d
from bitforge.training.surrogates import SignSTE, BinaryWeightSTE
from bitforge.training.bitnorm import BatchNormBN


class XNORBlock(nn.Module):
    """XNOR-Net BasicBlock: BinConv -> BN -> BinAct (-> repeat) + FP shortcut."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn0 = BatchNormBN(in_planes)
        self.sign0 = SignSTE(clip_val=1.5)
        self.conv1 = BinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, scale=True)
        self.bn1 = BatchNormBN(planes)
        self.sign1 = SignSTE(clip_val=1.5)
        self.conv2 = BinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, scale=True)
        self.bn2 = BatchNormBN(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                # FP 1x1 conv (XNOR-Net keeps shortcuts in FP for stability)
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.sign0(self.bn0(x))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.sign1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        return out


class XNORNet(nn.Module):
    """XNOR-Net for CIFAR/MNIST (FP first conv + binary body + FP last FC)."""

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
        # FP first conv (per XNOR-Net paper)
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = self._make_layer(widths[0], blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(widths[1], blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(widths[2], blocks_per_stage, stride=2)
        self.fc = nn.Linear(widths[2], num_classes)
        # mark FP modules for storage estimation
        for m in [self.conv1, self.fc]:
            m._bit_width = 32

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(XNORBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
