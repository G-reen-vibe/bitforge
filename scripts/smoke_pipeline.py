"""Quick end-to-end smoke test of the full training pipeline.

Uses a TINY subset (1000 train, 500 test) and small model to verify the
pipeline works without burning all CPU. Run with:

    python scripts/smoke_pipeline.py
"""
from __future__ import annotations

import os
import sys
import time

# path setup
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
torch.set_num_threads(2)

from torch.utils.data import Subset

from bitforge.utils.config import load_config
from bitforge.data.datasets import build_dataset, DATASET_INFO
from bitforge.models import build_model
from bitforge.training import Trainer
from bitforge.utils.config import Config
from bitforge.utils.seed import set_seed


def tiny_run(model_name: str, n_train: int = 1000, n_test: int = 500, epochs: int = 1):
    set_seed(0)
    cfg = load_config(os.path.join(HERE, "..", "configs", "mnist", "default.yaml"))
    cfg.set("model.name", model_name)
    # tiny model kwargs
    if model_name == "fp_resnet":
        cfg.set("model.base_width", 4)
        cfg.set("model.blocks_per_stage", 1)
    elif model_name == "hyper_lut":
        cfg.set("model.hv_dim", 256)
        cfg.set("model.k", 4)
        cfg.set("model.num_luts", 8)
        cfg.set("model.num_blocks", 2)
    else:
        cfg.set("model.base_width", 8)
        cfg.set("model.blocks_per_stage", 1)

    info = DATASET_INFO["mnist"]
    train_ds = build_dataset("mnist", root="./data", train=True, augment="none")
    test_ds = build_dataset("mnist", root="./data", train=False, augment="none")
    train_subset = Subset(train_ds, list(range(n_train)))
    test_subset = Subset(test_ds, list(range(n_test)))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)

    # build model
    model_cfg = Config(raw=cfg.to_dict())
    model_cfg.raw["model"].update({
        "in_channels": info.in_channels,
        "num_classes": info.num_classes,
        "img_size": info.img_size,
    })
    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        device="cpu",
        exp_name=f"smoke.{model_name}",
        seed=0,
        output_dir="./results",
        log_every=5,
        num_threads=2,
    )
    t0 = time.time()
    result = trainer.fit(epochs=epochs)
    dt = time.time() - t0
    return model_name, n_params, result.final.get("top1", 0.0), dt


if __name__ == "__main__":
    print(f"{'model':<15} {'params':>10} {'top1':>8} {'time_s':>8}")
    print("-" * 45)
    for m in ["fp_resnet", "reactnet", "hyper_lut"]:
        try:
            name, p, top1, dt = tiny_run(m, n_train=1000, n_test=500, epochs=1)
            print(f"{name:<15} {p:>10,} {top1:>8.4f} {dt:>8.1f}")
        except Exception as e:
            print(f"{m:<15} FAILED: {e}")
            import traceback; traceback.print_exc()
