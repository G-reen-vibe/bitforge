"""Experiment runner: build model+optimizer+dataloaders from Config, train,
and return a RunResult. Supports multi-seed sweeps with aggregation.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from bitforge.data.datasets import build_loaders, DATASET_INFO
from bitforge.models import build_model
from bitforge.training import Trainer, build_scheduler
from bitforge.training.surrogates import SURROGATES
from bitforge.utils.config import Config, load_config
from bitforge.utils.logging import get_logger, write_json
from bitforge.utils.metrics import RunResult, aggregate_runs
from bitforge.utils.seed import set_seed


def build_exp_name(model_name: str, dataset_name: str, tag: str = "") -> str:
    """Build a canonical experiment name like 'reactnet.cifar10' or 'hyper_lut.mnist.tag'."""
    base = f"{model_name}.{dataset_name}"
    return f"{base}.{tag}" if tag else base


def _build_optimizer(model: nn.Module, cfg: Config) -> optim.Optimizer:
    name = cfg.get("train.optimizer", "adam").lower()
    lr = float(cfg.get("train.lr", 1e-3))
    wd = float(cfg.get("train.weight_decay", 0.0))
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        m = float(cfg.get("train.momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def run_experiment(cfg: Config, seed: int, output_dir: str = "./results") -> RunResult:
    """Run a single training experiment given config + seed.

    Returns a RunResult with full history + best/final metrics.
    """
    set_seed(seed)
    dataset_name = cfg.get("data.name", "cifar10").lower()
    info = DATASET_INFO[dataset_name]
    batch_size = int(cfg.get("data.batch_size", 128))
    num_workers = int(cfg.get("data.num_workers", 0))

    train_loader, test_loader, info = build_loaders(
        dataset_name,
        root=cfg.get("data.root", "./data"),
        batch_size=batch_size,
        test_batch_size=int(cfg.get("data.test_batch_size", 256)),
        augment=cfg.get("data.augment", "light"),
        num_workers=num_workers,
        pin_memory=False,
        seed=seed,
        download=True,
    )

    # build model — inject dataset-specific args
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    model_kwargs.setdefault("in_channels", info.in_channels)
    model_kwargs.setdefault("num_classes", info.num_classes)
    model_kwargs.setdefault("img_size", info.img_size)
    # Build a Config that shares the same raw dict (so model.name is preserved)
    model_cfg = Config(raw=cfg.to_dict())
    model_cfg.raw["model"] = {**model_cfg.raw["model"], **model_kwargs}
    model = build_model(model_cfg)
    device = cfg.get("runtime.device", "cpu")
    model = model.to(device)

    optimizer = _build_optimizer(model, cfg)
    epochs = int(cfg.get("train.epochs", 100))
    scheduler = build_scheduler(
        optimizer,
        name=cfg.get("train.scheduler", "cosine"),
        T_max=epochs,
        eta_min=float(cfg.get("train.lr_min", 0.0)),
        milestones=cfg.get("train.milestones", [60, 120]),
        gamma=cfg.get("train.gamma", 0.1),
        step_size=cfg.get("train.step_size", 30),
        epochs=epochs,
    )

    # experiment name
    model_name = cfg.get("model.name", "hyper_lut")
    tag = cfg.get("runtime.tag", "")
    exp_name = build_exp_name(model_name, dataset_name, tag)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        exp_name=exp_name,
        seed=seed,
        output_dir=output_dir,
        log_every=int(cfg.get("runtime.log_every", 50)),
        grad_clip=float(cfg.get("train.grad_clip", 0.0)),
        time_budget_s=float(cfg.get("runtime.time_budget_s", 0.0)),
        num_threads=int(cfg.get("runtime.num_threads", 0)) or None,
    )

    # Hook for IR-Net EDE: anneal surrogate slope over epochs
    if hasattr(model, "set_slope"):
        ede_start = float(cfg.get("train.ede_start_slope", 2.0))
        ede_end = float(cfg.get("train.ede_end_slope", 8.0))
        _patch_trainer_for_ede(trainer, model, ede_start, ede_end, epochs)

    # Hook for Hyper-LUT: anneal Gumbel temperature
    if hasattr(model, "set_temperature") and hasattr(model, "get_temperature_schedule"):
        _patch_trainer_for_temp(trainer, model, epochs)

    result = trainer.fit(epochs=epochs)
    return result


def run_multi_seed(cfg: Config, seeds: List[int], output_dir: str = "./results") -> Dict[str, Any]:
    """Run `run_experiment` for each seed; aggregate; write a summary JSON."""
    logger = get_logger("runner")
    results: List[RunResult] = []
    for s in seeds:
        logger.info(f"=== seed {s} ===")
        r = run_experiment(cfg, seed=s, output_dir=output_dir)
        results.append(r)
    agg = aggregate_runs(results)
    summary = {
        "config": cfg.to_dict(),
        "seeds": seeds,
        "aggregated": agg,
        "runs": [asdict(r) for r in results],
    }
    model_name = cfg.get("model.name", "hyper_lut")
    dataset_name = cfg.get("data.name", "cifar10")
    tag = cfg.get("runtime.tag", "")
    exp_name = build_exp_name(model_name, dataset_name, tag)
    summary_path = os.path.join(output_dir, "metrics", f"{exp_name}.summary.json")
    write_json(summary_path, summary)
    logger.info(f"Summary written to {summary_path}")
    return summary


# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
def _patch_trainer_for_ede(trainer: Trainer, model, start_slope: float, end_slope: float, total_epochs: int):
    """Monkey-patch trainer._log_epoch to also update IR-Net's surrogate slope."""
    orig = trainer._log_epoch
    def new_log(epoch, train_m, test_m):
        # linearly anneal slope from start to end
        progress = (epoch - 1) / max(1, total_epochs - 1)
        slope = start_slope + (end_slope - start_slope) * progress
        model.set_slope(slope)
        orig(epoch, train_m, test_m)
    trainer._log_epoch = new_log


def _patch_trainer_for_temp(trainer: Trainer, model, total_epochs: int):
    """Monkey-patch trainer._log_epoch to also update Hyper-LUT temperature."""
    orig = trainer._log_epoch
    def new_log(epoch, train_m, test_m):
        t = model.get_temperature_schedule(epoch, total_epochs)
        model.set_temperature(t)
        orig(epoch, train_m, test_m)
    trainer._log_epoch = new_log
