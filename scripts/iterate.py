"""Fast iteration runner for Hyper-LUT Net research rounds.

Usage:
    python scripts/iterate.py --round N --tag "description-of-change"

Trains on a fixed small MNIST subset (5000 train, 1000 test) for 5 epochs
and appends results to results/iterations.csv. Designed for ~30-60s per run.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
torch.set_num_threads(2)

from torch.utils.data import Subset, DataLoader

from bitforge.data.datasets import build_dataset, DATASET_INFO
from bitforge.models.hyper_lut import HyperLUTNet
from bitforge.training import Trainer
from bitforge.utils.seed import set_seed


# Fixed iteration config — DO NOT change between rounds (that's hyperparameter tuning)
ITER_TRAIN_N = 5000
ITER_TEST_N = 1000
ITER_EPOCHS = 5
ITER_BATCH = 128
ITER_LR = 1e-3
ITER_SEED = 0
DATA_ROOT = "./data"
RESULTS_CSV = os.path.join(HERE, "..", "results", "iterations.csv")


def run_round(round_num: int, tag: str, model_kwargs: dict | None = None,
              distill: bool = False, model_name: str = "hyper_lut") -> dict:
    """Run one iteration round. Returns a results dict."""
    set_seed(ITER_SEED)
    info = DATASET_INFO["mnist"]

    # Cache datasets on first run
    train_ds = build_dataset("mnist", root=DATA_ROOT, train=True, augment="none")
    test_ds = build_dataset("mnist", root=DATA_ROOT, train=False, augment="none")
    train_subset = Subset(train_ds, list(range(ITER_TRAIN_N)))
    test_subset = Subset(test_ds, list(range(ITER_TEST_N)))
    train_loader = DataLoader(train_subset, batch_size=ITER_BATCH, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=ITER_BATCH, shuffle=False)

    # Default model kwargs — depend on which model
    if model_name == "binary_vit":
        from bitforge.models.binary_vit import BinaryViT
        defaults = dict(
            img_size=info.img_size,
            in_channels=info.in_channels,
            num_classes=info.num_classes,
            patch_size=4,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            ternary_mlp=False,
        )
    else:
        defaults = dict(
            in_channels=info.in_channels,
            num_classes=info.num_classes,
            img_size=info.img_size,
            hv_dim=512,
            k=4,
            num_luts=16,
            num_blocks=2,
            encoder_seed=42,
            binarize_input=False,
        )
    if model_kwargs:
        defaults.update(model_kwargs)

    if model_name == "binary_vit":
        model = BinaryViT(**defaults)
    else:
        model = HyperLUTNet(**defaults)
    n_params = sum(p.numel() for p in model.parameters())

    optim = torch.optim.Adam(model.parameters(), lr=ITER_LR)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        device="cpu",
        exp_name=f"iter.r{round_num}",
        seed=ITER_SEED,
        output_dir="./results",
        log_every=20,
        num_threads=2,
    )

    # Distillation: train a tiny FP teacher first, then add KD loss
    if distill:
        from bitforge.models.baselines import FPResNet
        teacher = FPResNet(in_channels=info.in_channels, num_classes=info.num_classes,
                           img_size=info.img_size, base_width=8, blocks_per_stage=1)
        teacher_optim = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        teacher_trainer = Trainer(
            model=teacher, train_loader=train_loader, test_loader=test_loader,
            optimizer=teacher_optim, device="cpu", exp_name=f"iter.r{round_num}.teacher",
            seed=ITER_SEED, output_dir="./results", log_every=100, num_threads=2,
        )
        print("Training teacher...")
        teacher_trainer.fit(epochs=ITER_EPOCHS)
        teacher.eval()
        # Patch training loop to use KD loss
        import torch.nn.functional as Fte
        import time as _time
        def _patched_train_epoch(epoch):
            model.train()
            t0 = _time.time()
            total_loss = 0.0
            total_correct = 0
            total_n = 0
            for step, (xb, yb) in enumerate(trainer.train_loader, 1):
                xb = xb.to(trainer.device); yb = yb.to(trainer.device)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                with torch.no_grad():
                    teacher_logits = teacher(xb)
                # KD loss: 0.5 * CE(student, y) + 0.5 * KL(student || teacher/2)
                ce = Fte.cross_entropy(logits, yb)
                kd = Fte.kl_div(Fte.log_softmax(logits / 2.0, dim=1),
                                 Fte.softmax(teacher_logits / 2.0, dim=1),
                                 reduction="batchmean") * 4.0
                loss = 0.5 * ce + 0.5 * kd
                loss.backward()
                optim.step()
                bs = yb.size(0)
                total_loss += loss.item() * bs
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += bs
                trainer.global_step += 1
                if step % trainer.log_every == 0 or step == len(trainer.train_loader):
                    trainer.logger.info(
                        f"[ep {epoch} step {step}/{len(trainer.train_loader)}] "
                        f"loss={loss.item():.4f} acc={total_correct/total_n:.4f} "
                        f"t={_time.time()-trainer.start_time:.0f}s"
                    )
            dt = _time.time() - t0
            return {"loss": total_loss/max(1,total_n), "top1": total_correct/max(1,total_n), "time_s": dt}
        trainer._train_epoch = _patched_train_epoch

    # If the model has a regularization_loss AND not distilling, monkey-patch
    # the trainer's _train_epoch to add it to the loss.
    if hasattr(model, "regularization_loss") and not distill:
        _orig_train_epoch = trainer._train_epoch
        def _patched_train_epoch(epoch):
            # we need to re-run the loop with the extra loss — easiest is to
            # patch the model forward to add the reg loss to a buffer attribute
            # that we read after backward. But to keep it simple, we just
            # add the reg loss to each batch's loss inside the loop.
            # Reimplement the loop here.
            model.train()
            import time as _time
            t0 = _time.time()
            total_loss = 0.0
            total_correct = 0
            total_n = 0
            for step, (xb, yb) in enumerate(trainer.train_loader, 1):
                xb = xb.to(trainer.device); yb = yb.to(trainer.device)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = trainer.criterion(logits, yb) + 0.01 * model.regularization_loss()
                loss.backward()
                optim.step()
                bs = yb.size(0)
                total_loss += loss.item() * bs
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += bs
                trainer.global_step += 1
                if step % trainer.log_every == 0 or step == len(trainer.train_loader):
                    trainer.logger.info(
                        f"[ep {epoch} step {step}/{len(trainer.train_loader)}] "
                        f"loss={loss.item():.4f} acc={total_correct/total_n:.4f} "
                        f"t={_time.time()-trainer.start_time:.0f}s"
                    )
            dt = _time.time() - t0
            return {"loss": total_loss/max(1,total_n), "top1": total_correct/max(1,total_n), "time_s": dt}
        trainer._train_epoch = _patched_train_epoch

    t0 = time.time()
    try:
        result = trainer.fit(epochs=ITER_EPOCHS)
        dt = time.time() - t0
        row = {
            "round": round_num,
            "tag": tag,
            "params": n_params,
            "best_top1": result.best.get("top1", 0.0),
            "final_top1": result.final.get("top1", 0.0),
            "final_loss": result.final.get("loss", 0.0),
            "time_s": round(dt, 1),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        dt = time.time() - t0
        row = {
            "round": round_num,
            "tag": tag,
            "params": n_params,
            "best_top1": 0.0,
            "final_top1": 0.0,
            "final_loss": 0.0,
            "time_s": round(dt, 1),
            "status": "fail",
            "error": str(e)[:200],
        }
        traceback.print_exc()
    return row


def append_csv(row: dict) -> None:
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--hv-dim", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--num-luts", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--multi-k", action="store_true", default=False)
    p.add_argument("--no-learnable-encoder", action="store_true", default=False)
    p.add_argument("--distill", action="store_true", default=False)
    p.add_argument("--spatial", action="store_true", default=False)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--model", type=str, default="hyper_lut", choices=["hyper_lut", "binary_vit"])
    args = p.parse_args()

    mk = {}
    if args.hv_dim is not None: mk["hv_dim"] = args.hv_dim
    if args.k is not None: mk["k"] = args.k
    if args.num_luts is not None: mk["num_luts"] = args.num_luts
    if args.num_blocks is not None: mk["num_blocks"] = args.num_blocks
    if args.multi_k: mk["multi_k"] = True
    if args.no_learnable_encoder: mk["learnable_encoder"] = False
    if args.spatial: mk["spatial_encoder"] = True
    if args.drop_path > 0: mk["drop_path"] = args.drop_path

    print(f"\n=== ROUND {args.round}: {args.tag} ===")
    row = run_round(args.round, args.tag, model_kwargs=mk or None,
                    distill=args.distill, model_name=args.model)
    append_csv(row)
    print(f"\n=== RESULT: top1={row['best_top1']:.4f} loss={row['final_loss']:.4f} "
          f"time={row['time_s']}s status={row['status']} ===")
    if row["status"] != "ok":
        print(f"ERROR: {row['error']}")


if __name__ == "__main__":
    main()
