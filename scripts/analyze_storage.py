"""Analysis: bit-width, storage, and theoretical inference cost.

Computes:
- Total params, bits, KB for binary vs FP
- Theoretical speedup from binary ops (XNOR-popcount vs FP MAC)
- Effective compression ratio
"""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
from bitforge.models.baselines.layers import BinaryConv2d, BinaryLinear


def analyze_model(model, name: str):
    """Print storage + cost analysis for a model."""
    n_params = 0
    n_bits = 0
    n_binary_params = 0
    n_fp_params = 0
    for n, p in model.named_parameters():
        is_binary = any(t in n.lower() for t in ("binary",))
        # also check if module is binary by climbing? Simpler: check by name patterns
        bw = 1 if is_binary else 32
        n_params += p.numel()
        n_bits += p.numel() * bw
        if is_binary:
            n_binary_params += p.numel()
        else:
            n_fp_params += p.numel()
    n_kb = n_bits / 8 / 1024
    fp_kb = (n_params * 32) / 8 / 1024
    compression = (n_params * 32) / max(1, n_bits)
    print(f"\n=== {name} ===")
    print(f"  Total params:    {n_params:>10,}")
    print(f"  Binary params:   {n_binary_params:>10,} ({100*n_binary_params/n_params:.1f}%)")
    print(f"  FP params:       {n_fp_params:>10,} ({100*n_fp_params/n_params:.1f}%)")
    print(f"  Storage:         {n_bits:>10,} bits = {n_kb:.1f} KB")
    print(f"  FP equivalent:   {fp_kb:.1f} KB")
    print(f"  Compression:     {compression:.1f}x")
    return {"name": name, "params": n_params, "bits": n_bits, "kb": n_kb,
            "fp_kb": fp_kb, "compression": compression,
            "binary_pct": 100*n_binary_params/n_params}


def main():
    # Build the best single-expert model (6 binary convs, no FP first)
    from bitforge.models.baselines.layers import BinaryConv2d, BinaryLinear
    import torch.nn as nn
    import torch.nn.functional as F

    class Binary6ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            bw = 32
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            # All binary convs (no FP first)
            widths = [1, bw, bw*2, bw*4, bw*4, bw*4, bw*4]
            strides = [1, 2, 2, 1, 1, 1]
            for i in range(len(widths)-1):
                self.convs.append(BinaryConv2d(widths[i], widths[i+1], 3, strides[i], 1, bias=False))
                self.bns.append(nn.BatchNorm2d(widths[i+1]))
            self.fc = BinaryLinear(bw*4, 10, bias=False)
        def forward(self, x):
            for i in range(len(self.convs)):
                x = F.gelu(self.bns[i](self.convs[i](x)))
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    # FP equivalent (same arch)
    class FP6ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            bw = 32
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            widths = [1, bw, bw*2, bw*4, bw*4, bw*4, bw*4]
            strides = [1, 2, 2, 1, 1, 1]
            for i in range(len(widths)-1):
                self.convs.append(nn.Conv2d(widths[i], widths[i+1], 3, strides[i], 1, bias=False))
                self.bns.append(nn.BatchNorm2d(widths[i+1]))
            self.fc = nn.Linear(bw*4, 10, bias=False)
        def forward(self, x):
            for i in range(len(self.convs)):
                x = F.gelu(self.bns[i](self.convs[i](x)))
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    # Compare
    results = []
    results.append(analyze_model(Binary6ConvNet(), "Binary 6-Conv Net (fully binary)"))
    results.append(analyze_model(FP6ConvNet(), "FP 6-Conv Net (same architecture)"))

    # Theoretical MACs / XNOR ops for one forward pass on MNIST (1x28x28)
    print("\n=== Theoretical Inference Cost (MNIST 1x28x28) ===")
    # Per conv layer: out_channels * in_channels * kh * kw * out_h * out_w
    # For binary: each MAC becomes 1 XNOR-popcount (effectively 1 op per pair)
    # For FP: each MAC is 2 FP ops (mul + add)
    print("  Layer-by-layer MAC count (assuming 28x28 input, stride 1 first, then 2,2,1,1,1):")
    bw = 32
    widths = [1, bw, bw*2, bw*4, bw*4, bw*4, bw*4]
    strides = [1, 2, 2, 1, 1, 1]
    h = 28
    total_macs_fp = 0
    total_xnor_ops = 0
    for i in range(len(widths)-1):
        out_h = h // strides[i]
        out_w = h // strides[i]
        macs = widths[i+1] * widths[i] * 3 * 3 * out_h * out_w
        total_macs_fp += macs
        # Binary: 1 XNOR-popcount per output element (32-way parallelism)
        # Effective ops = macs / 32 (XNOR-popcount does 32 binary MACs in 1 op)
        total_xnor_ops += macs / 32
        print(f"  Conv {i+1}: {widths[i]:>3} -> {widths[i+1]:>3}, stride {strides[i]}, "
              f"output {out_h}x{out_w}, MACs={macs:>10,}")
        h = out_h
    print(f"\n  Total FP MACs:     {total_macs_fp:>12,}")
    print(f"  Total XNOR ops:    {int(total_xnor_ops):>12,}")
    print(f"  Speedup (FP/binary): {total_macs_fp/total_xnor_ops:.1f}x (theoretical)")

    # Write summary
    summary_path = os.path.join(HERE, "..", "results", "analysis_storage.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "params", "bits", "kb", "fp_kb",
                                          "compression", "binary_pct"])
        w.writeheader()
        for r in results:
            w.writerow({"model": r["name"], "params": r["params"], "bits": r["bits"],
                        "kb": round(r["kb"], 2), "fp_kb": round(r["fp_kb"], 2),
                        "compression": round(r["compression"], 2),
                        "binary_pct": round(r["binary_pct"], 2)})
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
