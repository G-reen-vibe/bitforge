"""BinaryMoE: Mixture of Tiny Binary CNN Experts.

Completely different paradigm from both Hyper-LUT and BinaryViT:
    - Multiple tiny binary CNN experts (each ~5K params)
    - Learned gate picks top-2 experts per sample (sparse)
    - Weighted average of expert logits for final prediction
    - Only top-2 experts compute = inference efficiency
    - Each expert can specialize (e.g., one per class cluster)

This is rounds 51-75 of the iteration plan.
"""
from bitforge.models.binary_moe.expert import TinyBinaryExpert
from bitforge.models.binary_moe.gate import TopKGate
from bitforge.models.binary_moe.model import BinaryMoE

__all__ = ["TinyBinaryExpert", "TopKGate", "BinaryMoE"]
