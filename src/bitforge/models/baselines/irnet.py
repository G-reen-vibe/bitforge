"""IR-Net (Qin et al., 2020): "Balanced and Information-Retentive" binary net.

Key contributions reproduced here:
    - Libra Parameter Binarization (Libra-PB): balances weight distribution
      before binarization (subtract mean) and uses an unbiased ATan surrogate
      gradient.
    - Error Decay Estimator (EDE): anneals the surrogate's slope over training
      to match the discrete sign function at the end of training.

For simplicity, the slope annealing is controlled by the Trainer via a hook
(`update_surrogate_slope`); the model exposes a `set_slope(s)` method.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitforge.models.baselines.layers import BinaryConv2d
from bitforge.training.surrogates import ATanSurrogate
from bitforge.training.bitnorm import BatchNormBN


class LibraPB(nn.Module):
    """Libra Parameter Binarization: subtract mean then sign, with ATan STE.

    Forward:  w_b = sign(w - mean(w)) * (mean(|w - mean(w)|) per-channel)
    Backward: atan surrogate with current slope.
    """

    def __init__(self, num_features: int):
        super().__init__()
        # we'll re-use ATanSurrogate but adapt forward to be weight-binarization
        self.slope = 2.0  # updated by Trainer via set_slope

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w shape: (out, in, kh, kw) for conv or (out, in) for linear
        dims = tuple(range(1, w.dim()))
        mean = w.mean(dim=dims, keepdim=True)
        w_centered = w - mean
        scale = w_centered.abs().mean(dim=dims, keepdim=True)
        wb = torch.sign(w_centered) * scale
        if self.training:
            # atan surrogate grad (slope-managed externally)
            s = self.slope
            grad = s / (1.0 + (3.14159265 * s * w_centered / 2.0) ** 2)
            return wb + (w_centered - w_centered.detach()) * grad
        return wb


class IRBlock(nn.Module):
    """IR-Net BasicBlock."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn0 = BatchNormBN(in_planes)
        self.sign0 = ATanSurrogate(clip_val=1.5, slope=2.0)
        self.conv1 = BinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, scale=False)
        self.bn1 = BatchNormBN(planes)
        self.sign1 = ATanSurrogate(clip_val=1.5, slope=2.0)
        self.conv2 = BinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, scale=False)
        self.bn2 = BatchNormBN(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
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


class IRNet(nn.Module):
    """IR-Net for CIFAR/MNIST."""

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
        # collect all surrogate modules for slope annealing
        self._surrogates = [m for m in self.modules() if isinstance(m, ATanSurrogate)]

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(IRBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def set_slope(self, slope: float) -> None:
        """EDE: update all surrogate slopes (called by Trainer per epoch)."""
        for m in self._surrogates:
            m.slope = slope

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
