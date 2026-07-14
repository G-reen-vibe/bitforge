"""Baseline models: FP-ResNet, XNOR-Net, Bi-Real Net, IR-Net, ReActNet.

Each model is parametrized by:
    - in_channels, num_classes, img_size (auto-fit)
    - width_multiplier (channel scaling)
    - depth: 'small' (for MNIST) or 'standard' (for CIFAR/ImageNet)

Design notes:
    - All baselines share a common ResNet-style block for fair comparison.
    - Differences are isolated to (a) the surrogate used for activations and
      weights, (b) the presence of Bi-Real's real-valued first/last layers,
      (c) ReActNet's learnable RSign/RPreF, (d) IR-Net's atan surrogate.
    - FP-ResNet is the upper-bound baseline; uses standard Conv+BN+ReLU.
"""
from bitforge.models.baselines.fp_resnet import FPResNet
from bitforge.models.baselines.xnor_net import XNORNet
from bitforge.models.baselines.bireal import BiRealNet
from bitforge.models.baselines.irnet import IRNet
from bitforge.models.baselines.reactnet import ReActNet

BASELINE_REGISTRY = {
    "fp_resnet": FPResNet,
    "xnor_net": XNORNet,
    "bireal": BiRealNet,
    "irnet": IRNet,
    "reactnet": ReActNet,
}

__all__ = ["FPResNet", "XNORNet", "BiRealNet", "IRNet", "ReActNet", "BASELINE_REGISTRY"]
