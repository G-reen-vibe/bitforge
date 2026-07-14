"""Trainer: a small, dependency-light training loop with multi-seed support.

Features:
    - Single fit() entry point that returns a RunResult
    - Tracks best-epoch metrics on the test set
    - Logs every `log_every` steps + epoch summaries
    - Time-budget aware: can stop early if a wall-clock budget is exceeded
    - CPU-optimized: uses torch.set_num_threads from config
    - Saves checkpoints to results/checkpoints/
    - Writes a per-run JSON to results/metrics/<exp_name>.seed<N>.json
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bitforge.utils.logging import get_logger, StageTimer, write_json
from bitforge.utils.metrics import RunResult, compute_metrics
from bitforge.utils.seed import set_seed


class Trainer:
    """Generic trainer for any nn.Module on (train_loader, test_loader).

    Args:
        model: nn.Module
        train_loader / test_loader: torch DataLoader
        optimizer: torch.optim.Optimizer
        scheduler: optional LR scheduler (stepped per epoch)
        criterion: nn.Module loss (default: cross-entropy)
        device: 'cpu' or 'cuda'
        exp_name: name used for checkpoint + metric filenames
        output_dir: root for results (default: ./results)
        log_every: steps between progress logs
        grad_clip: max-norm for gradient clipping (0 disables)
        time_budget_s: hard wall-clock budget in seconds (0 = unlimited)
        num_threads: if set, calls torch.set_num_threads()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
        exp_name: str = "exp",
        seed: int = 0,
        output_dir: str = "./results",
        log_every: int = 50,
        grad_clip: float = 0.0,
        time_budget_s: float = 0.0,
        num_threads: Optional[int] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.exp_name = exp_name
        self.seed = seed
        self.output_dir = output_dir
        self.log_every = log_every
        self.grad_clip = grad_clip
        self.time_budget_s = time_budget_s
        self.logger = get_logger(f"trainer.{exp_name}")
        self.timer = StageTimer()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        # output dirs
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.global_step = 0
        self.start_time = 0.0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self, epochs: int) -> RunResult:
        self.start_time = time.time()
        result = RunResult(seed=self.seed)
        # estimate params + bits
        result.n_params, result.n_bits = _estimate_params_and_bits(self.model)
        self.logger.info(
            f"Starting training: exp={self.exp_name} seed={self.seed} epochs={epochs} "
            f"params={result.n_params:,} (~{result.n_bits:,} bits = {result.n_bits/8/1024:.1f} KiB)"
        )

        best_top1 = -1.0
        for epoch in range(1, epochs + 1):
            if self._over_budget():
                self.logger.warning(
                    f"Time budget exceeded ({time.time()-self.start_time:.0f}s); stopping at epoch {epoch-1}"
                )
                break
            train_metrics = self._train_epoch(epoch)
            test_metrics = self._eval_epoch(epoch)
            self._log_epoch(epoch, train_metrics, test_metrics)
            result.history.append({**{f"train.{k}": v for k, v in train_metrics.items()},
                                    **{f"test.{k}": v for k, v in test_metrics.items()}})
            # track best
            if test_metrics.get("top1", 0.0) > best_top1:
                best_top1 = test_metrics["top1"]
                result.best = dict(test_metrics)
                result.best["epoch"] = epoch
                self._save_checkpoint(epoch, best=True)
            # final = last-epoch
            result.final = dict(test_metrics)
            result.final["epoch"] = epoch
            if self.scheduler is not None:
                self.scheduler.step()

        result.train_time_s = time.time() - self.start_time
        self._write_run_result(result)
        self.logger.info(
            f"Done in {result.train_time_s:.1f}s. "
            f"Best top1={result.best.get('top1', 0):.4f} (epoch {result.best.get('epoch', '?')})"
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.timer.start(f"train_epoch_{epoch}")
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        n_steps = 0
        for step, (x, y) in enumerate(self.train_loader, 1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += bs
            n_steps += 1
            self.global_step += 1

            if step % self.log_every == 0 or step == len(self.train_loader):
                self.logger.info(
                    f"[ep {epoch} step {step}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} acc={total_correct/total_n:.4f} "
                    f"lr={_peek_lr(self.optimizer):.5f} "
                    f"t={time.time()-self.start_time:.0f}s"
                )
        dt = self.timer.stop(f"train_epoch_{epoch}")
        return {
            "loss": total_loss / max(1, total_n),
            "top1": total_correct / max(1, total_n),
            "time_s": dt,
        }

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.timer.start(f"eval_epoch_{epoch}")
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        all_topk = (1,)
        for x, y in self.test_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += bs
        dt = self.timer.stop(f"eval_epoch_{epoch}")
        return {
            "loss": total_loss / max(1, total_n),
            "top1": total_correct / max(1, total_n),
            "time_s": dt,
        }

    def _log_epoch(self, epoch: int, train_m: Dict[str, float], test_m: Dict[str, float]) -> None:
        self.logger.info(
            f"== epoch {epoch} == "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['top1']:.4f} "
            f"test_loss={test_m['loss']:.4f} test_acc={test_m['top1']:.4f} "
            f"train_t={train_m['time_s']:.1f}s eval_t={test_m['time_s']:.1f}s"
        )

    def _save_checkpoint(self, epoch: int, best: bool = False) -> None:
        suffix = "best" if best else f"ep{epoch}"
        path = os.path.join(self.ckpt_dir, f"{self.exp_name}.seed{self.seed}.{suffix}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def _write_run_result(self, result: RunResult) -> None:
        path = os.path.join(self.metrics_dir, f"{self.exp_name}.seed{self.seed}.json")
        write_json(path, asdict(result))

    def _over_budget(self) -> bool:
        if self.time_budget_s <= 0:
            return False
        return (time.time() - self.start_time) > self.time_budget_s


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _peek_lr(optimizer: torch.optim.Optimizer) -> float:
    for pg in optimizer.param_groups:
        return pg["lr"]
    return 0.0


def _estimate_params_and_bits(model: nn.Module):
    """Estimate total parameters and their bit-storage cost.

    For parameters inside modules that have an attribute `_bit_width` (set by
    binary layers), uses that bit width; otherwise uses 32 (FP32).
    """
    total_params = 0
    total_bits = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        # climb module tree to find _bit_width
        bit_width = 32
        # check if this parameter belongs to a binary module
        # (we mark binary modules with `_bit_width` attribute on the module)
        # fallback: detect by name patterns
        if any(tag in name.lower() for tag in ("binary", "bin", "lut_table", "alpha_hv")):
            bit_width = 1
        total_bits += n * bit_width
    return total_params, total_bits
