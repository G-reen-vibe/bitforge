"""BinaryViT: A 1-bit Vision Transformer built from scratch.

Completely different paradigm from Hyper-LUT Net:
    - Patch-based: image -> N patch tokens (preserves spatial structure)
    - Binary self-attention: Q, K, V via binary linear (XNOR-popcount)
    - Ternary weights {-1, 0, +1} for sparsity (2 bits per weight)
    - Binary-friendly LayerNorm via sign-shift
    - Polynomial surrogate gradient (no Gumbel, no STE-on-sign alone)

This is rounds 26-50 of the iteration plan.
"""
from bitforge.models.binary_vit.components import (
    BinaryLinear,
    TernaryLinear,
    BinaryPatchEmbed,
    BinarySelfAttention,
    BinaryMLP,
    BinaryViTBlock,
    BinaryLayerNorm,
)
from bitforge.models.binary_vit.model import BinaryViT

__all__ = [
    "BinaryLinear", "TernaryLinear", "BinaryPatchEmbed",
    "BinarySelfAttention", "BinaryMLP", "BinaryViTBlock",
    "BinaryLayerNorm", "BinaryViT",
]
