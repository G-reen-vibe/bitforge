"""BinaryViT model: patch embed -> N blocks -> mean-pool -> binary classifier."""
from __future__ import annotations

import torch
import torch.nn as nn

from bitforge.models.binary_vit.components import (
    BinaryPatchEmbed, BinaryViTBlock, BinaryLinear, BinaryLayerNorm,
)


class BinaryViT(nn.Module):
    """A 1-bit Vision Transformer.

    Args:
        img_size: square image size
        patch_size: patch size for embedding
        in_channels: 1 for MNIST, 3 for CIFAR
        num_classes: classification head
        embed_dim: D, patch embedding dim
        depth: number of ViT blocks
        num_heads: attention heads
        mlp_ratio: MLP hidden dim multiplier
        dropout: dropout rate (0 disables)
        ternary_classifier: if True, use ternary weights for the classifier head
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        ternary_classifier: bool = False,
        ternary_mlp: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        # Binary conv stem: 2 FP conv layers to extract local features
        # before patchification. This is the "early conv" trick from
        # Xiao et al. (Early Convolutions Help Transformers See Better).
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.patch_embed = BinaryPatchEmbed(img_size, patch_size, embed_dim, embed_dim)
        num_patches = self.patch_embed.num_patches
        # learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            BinaryViTBlock(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout,
                           ternary_mlp=ternary_mlp)
            for _ in range(depth)
        ])
        self.norm = BinaryLayerNorm(embed_dim)
        if ternary_classifier:
            from bitforge.models.binary_vit.components import TernaryLinear
            self.fc = TernaryLinear(embed_dim, num_classes, bias=True)
        else:
            self.fc = BinaryLinear(embed_dim, num_classes, bias=True)
        self.register_buffer("logit_scale", torch.tensor(1.0 / (embed_dim ** 0.5)))
        for m in [self.patch_embed, self.fc] + list(self.blocks):
            m._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)  # (N, embed_dim, H, W) — FP local features
        x = self.patch_embed(x)  # (N, L, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # mean-pool over patches
        x = x.mean(dim=1)  # (N, D)
        return self.logit_scale * self.fc(x)
