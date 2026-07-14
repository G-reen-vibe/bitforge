"""Smoke tests: ensure every model builds and runs a forward pass on a tiny input."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitforge.models.baselines import (
    FPResNet, XNORNet, BiRealNet, IRNet, ReActNet, BASELINE_REGISTRY
)
from bitforge.models.hyper_lut import HyperLUTNet


TINY_MODELS = [
    ("fp_resnet", dict(in_channels=3, num_classes=10, img_size=32, base_width=8, blocks_per_stage=1)),
    ("xnor_net",  dict(in_channels=3, num_classes=10, img_size=32, base_width=8, blocks_per_stage=1)),
    ("bireal",    dict(in_channels=3, num_classes=10, img_size=32, base_width=8, blocks_per_stage=1)),
    ("irnet",     dict(in_channels=3, num_classes=10, img_size=32, base_width=8, blocks_per_stage=1)),
    ("reactnet",  dict(in_channels=3, num_classes=10, img_size=32, base_width=8, blocks_per_stage=1)),
    ("hyper_lut", dict(in_channels=3, num_classes=10, img_size=32, hv_dim=256, k=4, num_luts=8, num_blocks=2)),
]


@pytest.mark.parametrize("name,kwargs", TINY_MODELS)
def test_model_forward(name, kwargs):
    """Each model must accept (N, C, H, W) input and return (N, num_classes)."""
    if name == "hyper_lut":
        model = HyperLUTNet(**kwargs)
    else:
        model = BASELINE_REGISTRY[name](**kwargs)
    model.train()
    model.set_slope(2.0) if hasattr(model, "set_slope") else None
    model.set_temperature(1.0) if hasattr(model, "set_temperature") else None
    x = torch.randn(2, kwargs["in_channels"], kwargs["img_size"], kwargs["img_size"])
    out = model(x)
    assert out.shape == (2, kwargs["num_classes"]), f"{name} output shape: {out.shape}"
    # backward
    out.sum().backward()
    # at least one parameter has a gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, f"{name}: no gradients after backward"


def test_model_eval_mode():
    """Models should run in eval mode (no STE) without error."""
    model = HyperLUTNet(in_channels=3, num_classes=10, img_size=32, hv_dim=128, k=4, num_luts=4, num_blocks=1)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
