# BitForge

A 1-bit machine learning framework built from scratch, implementing **Hyper-LUT Net** — a novel architecture combining Hyperdimensional Computing (HDC) input encoding with differentiable LUT layers trained via Gumbel-softmax relaxation, **without using any Straight-Through Estimator (STE) for the main model**.

The repo also reproduces five SOTA baselines for fair comparison: FP-ResNet, XNOR-Net, Bi-Real Net, IR-Net, and ReActNet.

## Research Goal

Design a brand new 1-bit machine learning framework that:
1. Is **extremely efficient** in both training and inference
2. **Performs comparable to SOTA** on basic vision classification tasks (MNIST, CIFAR-10, CIFAR-100)
3. Avoids the known pathologies of STE-based BNNs (sign-mismatch, magnitude blindness, dead bits)
4. Embraces binary as the *native* representation rather than bolting binarization onto real-valued architectures

## Architecture: Hyper-LUT Net

```
Input (C, H, W)
  │
  ▼
HDC Encoder (random ±1 projection, fixed)
  │  → binary hypervector (N, D=2048)
  ▼
┌─────────────────────────────────────┐
│ LUT Block × N                       │
│  - Extract k=4-bit groups           │
│  - Differentiable LUT (2^k entries, │
│    Gumbel-softmax, annealed)        │
│  - Project + binarize back to D     │
│  - Permute (HDC orthogonal reorder) │
└─────────────────────────────────────┘
  │  → binary hypervector (N, D=2048)
  ▼
Binary Linear Readout (XNOR-popcount)
  │  → logits (N, num_classes)
```

**Why this is novel:**
- No STE anywhere — Gumbel-softmax is a principled relaxation with known convergence
- HDC encoding gives noise-robust, high-density input representation
- LUT layers give maximum boolean expressivity per bit (any function of k inputs)
- Inference is hardware-trivial: memory lookups + majority + XOR

## Repository Structure

```
bitforge/
├── configs/              # YAML configs per dataset
│   ├── base.yaml
│   ├── mnist/default.yaml
│   ├── cifar10/default.yaml
│   └── cifar100/default.yaml
├── src/bitforge/
│   ├── data/             # Dataset wrappers (MNIST, CIFAR-10/100)
│   ├── models/
│   │   ├── baselines/    # FP-ResNet, XNOR-Net, Bi-Real, IR-Net, ReActNet
│   │   └── hyper_lut/    # HDCEncoder, DifferentiableLUT, LUTBlock, HyperLUTNet
│   ├── training/         # Surrogates (STE/ATan/Polytanh/Swish), BitNorm, Trainer
│   ├── evaluation/       # Runner, multi-seed aggregation, Reporter
│   └── utils/            # Config, seeding, logging, metrics
├── experiments/          # CLI entry points
│   ├── run_baseline.py
│   ├── run_hyper_lut.py
│   └── run_sweep.py
├── tests/                # Smoke + unit tests (pytest)
└── results/              # Logs, checkpoints, metrics (gitignored)
```

## Installation

```bash
# CPU-only PyTorch (small footprint, ~200MB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Smoke test (single seed, 1 epoch, MNIST)

```bash
# FP-ResNet baseline
python experiments/run_baseline.py --model fp_resnet --dataset mnist --seeds 0 --epochs 1 --tag smoke

# ReActNet (SOTA binary CNN)
python experiments/run_baseline.py --model reactnet --dataset mnist --seeds 0 --epochs 1 --tag smoke

# Hyper-LUT Net
python experiments/run_hyper_lut.py --dataset mnist --seeds 0 --epochs 1 --tag smoke
```

### Multi-seed experiment (with error bars)

```bash
# 3 seeds of ReActNet on CIFAR-10
python experiments/run_baseline.py --model reactnet --dataset cifar10 --seeds 0,1,2
```

### Full sweep (all baselines × all datasets × N seeds)

```bash
# Quick smoke sweep on MNIST, 1 seed, 1 epoch
python experiments/run_sweep.py --datasets mnist --seeds 0 --epochs 1 --tag smoke

# Full CIFAR-10 sweep with 3 seeds
python experiments/run_sweep.py --datasets cifar10 --seeds 0,1,2
```

The sweep produces a Markdown report at `results/report.md` with mean ± std per experiment.

## Baselines Reproduced

| Model | Year | Key Mechanism | Use |
|-------|------|---------------|-----|
| **FP-ResNet** | 2015 | Standard FP32 ResNet | Upper-bound baseline |
| **XNOR-Net** | 2016 | Binary weights + binary activations + per-channel scaling | Classic BNN baseline |
| **Bi-Real Net** | 2018 | Real-activation shortcuts + clip-STE | Strong BNN baseline |
| **IR-Net** | 2020 | Libra-PB + ATan surrogate + EDE slope annealing | Strong BNN baseline |
| **ReActNet** | 2020 | RSign + RPreF (learnable shifts) | SOTA binary CNN |
| **Hyper-LUT Net** | (this work) | HDC encoder + Gumbel-LUT + BitNorm | Main contribution |

## Evaluation Protocol

- **Seeds**: All experiments support multi-seed runs. Default: 3 seeds (0, 1, 2). Reports include mean ± std.
- **Metrics**: Top-1 accuracy (best epoch + final epoch), loss, training time, parameter count, estimated bit storage.
- **Time budgets**: Each run has a wall-clock cap (configurable via `runtime.time_budget_s`).
- **Aggregation**: `run_multi_seed` writes `<exp_name>.summary.json` with aggregated stats. `Reporter` collates all summaries into `results/report.md`.

## Resource-Conscious Design

This codebase is designed for a **2-core CPU, 4GB RAM** environment:

- CPU-only PyTorch (no CUDA bloat)
- Small batch sizes (default 128)
- `num_workers=0` (we have only 2 cores; more workers thrash)
- Small model defaults (e.g., ResNet with base_width=16 for FP, 32 for BNNs)
- Time budgets per run (default 10min for MNIST, 1h for CIFAR)
- Smaller defaults than typical paper configs (epochs=30 for MNIST vs 100+ in papers)
- Skip-existing logic in `run_sweep.py` to resume interrupted sweeps

## Research Roadmap

- [x] **Stage 0**: Setup — environment, baselines, benchmarks, multi-seed eval (this commit)
- [ ] **Stage 1**: Smoke-test all baselines on MNIST (1-epoch validation)
- [ ] **Stage 2**: Full multi-seed baseline runs on MNIST + CIFAR-10
- [ ] **Stage 3**: Iterate on Hyper-LUT design (k, num_luts, num_blocks, temperature schedule)
- [ ] **Stage 4**: CIFAR-100 + scaling experiments
- [ ] **Stage 5**: Write-up of findings + ablations

## License

MIT

## Citation

```
@misc{bitforge2026,
  title={BitForge: A 1-bit machine learning framework built from scratch},
  author={G-reen-vibe},
  year={2026},
  url={https://github.com/G-reen-vibe/bitforge}
}
```
