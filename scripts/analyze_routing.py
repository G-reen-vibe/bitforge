"""Analysis: routing patterns and expert specialization in BinaryMoE.

Trains a BinaryMoE for 5 epochs, then records:
- Which expert handles which class
- Routing entropy (how concentrated)
- Per-expert accuracy on its routed samples
"""
import os
import sys
import csv
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
torch.set_num_threads(2)
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from bitforge.data.datasets import build_dataset, DATASET_INFO
from bitforge.training import Trainer
from bitforge.models.binary_moe import BinaryMoE
from bitforge.utils.seed import set_seed


def main():
    set_seed(0)
    train_ds = build_dataset("mnist", root="./data", train=True, augment="light")
    test_ds = build_dataset("mnist", root="./data", train=False, augment="none")
    train_loader = DataLoader(Subset(train_ds, list(range(5000))), batch_size=128, shuffle=True)
    test_loader = DataLoader(Subset(test_ds, list(range(1000))), batch_size=128, shuffle=False)

    model = BinaryMoE(in_channels=1, img_size=28, num_classes=10, num_experts=4, topk=2, base_width=32)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader,
                      optimizer=optim, device="cpu", exp_name="analysis.moe",
                      seed=0, output_dir="./results", log_every=100, num_threads=2)
    print("Training BinaryMoE for analysis...")
    trainer.fit(epochs=5)

    # Analyze routing on test set
    model.eval()
    num_experts = 4
    num_classes = 10
    # Per-class expert selection counts
    class_expert_counts = torch.zeros(num_classes, num_experts)
    # Per-expert correct count when selected
    expert_selected = torch.zeros(num_experts)
    expert_correct = torch.zeros(num_experts)
    # Routing entropy
    total_entropy = 0.0
    n_samples = 0
    # Per-expert per-class accuracy
    expert_class_correct = torch.zeros(num_experts, num_classes)
    expert_class_total = torch.zeros(num_experts, num_classes)

    with torch.no_grad():
        for x, y in test_loader:
            gates, expert_idx = model.gate(x)  # (N, E), (N, topk)
            expert_logits = torch.stack([e(x) for e in model.experts], dim=1)  # (N, E, C)
            # For each sample, record which experts were selected
            for i in range(x.size(0)):
                cls = y[i].item()
                selected = expert_idx[i].tolist()  # topk expert indices
                for ei in selected:
                    class_expert_counts[cls, ei] += 1
                    expert_selected[ei] += 1
                    pred = expert_logits[i, ei].argmax().item()
                    if pred == cls:
                        expert_correct[ei] += 1
                        expert_class_correct[ei, cls] += 1
                    expert_class_total[ei, cls] += 1
            # Routing entropy per sample
            p = gates + 1e-10
            ent = -(p * p.log()).sum(dim=1).mean().item()
            total_entropy += ent * x.size(0)
            n_samples += x.size(0)

    print(f"\n=== Routing Analysis (BinaryMoE, 4 experts, top-2) ===")
    print(f"Samples analyzed: {n_samples}")
    print(f"Mean routing entropy: {total_entropy/n_samples:.4f} bits (max = log2(4) = 2.0)")
    print(f"\nPer-expert selection rate:")
    for ei in range(num_experts):
        rate = expert_selected[ei].item() / n_samples
        acc = expert_correct[ei].item() / max(1, expert_selected[ei].item())
        print(f"  Expert {ei}: selected {rate*100:.1f}% of samples, "
              f"accuracy when selected = {acc*100:.2f}%")

    print(f"\nClass-Expert routing matrix (how often each expert handles each class):")
    print(f"{'Class':>6}", end="")
    for ei in range(num_experts):
        print(f"  Exp{ei}", end="")
    print()
    for cls in range(num_classes):
        total_for_class = class_expert_counts[cls].sum().item()
        print(f"  {cls:>3}  ", end="")
        for ei in range(num_experts):
            pct = 100 * class_expert_counts[cls, ei].item() / max(1, total_for_class)
            print(f"  {pct:5.1f}", end="")
        print()

    print(f"\nPer-expert per-class accuracy (when expert selected AND class = c):")
    print(f"{'Class':>6}", end="")
    for ei in range(num_experts):
        print(f"  Exp{ei}", end="")
    print()
    for cls in range(num_classes):
        print(f"  {cls:>3}  ", end="")
        for ei in range(num_experts):
            acc = 100 * expert_class_correct[ei, cls].item() / max(1, expert_class_total[ei, cls].item())
            print(f"  {acc:5.1f}", end="")
        print()

    # Write to CSV
    out_path = os.path.join(HERE, "..", "results", "analysis_routing.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mean_routing_entropy_bits", total_entropy/n_samples])
        w.writerow(["max_entropy_bits", 2.0])
        for ei in range(num_experts):
            rate = expert_selected[ei].item() / n_samples
            acc = expert_correct[ei].item() / max(1, expert_selected[ei].item())
            w.writerow([f"expert_{ei}_selection_rate", rate])
            w.writerow([f"expert_{ei}_accuracy_when_selected", acc])
    print(f"\nRouting analysis written to {out_path}")


if __name__ == "__main__":
    main()
