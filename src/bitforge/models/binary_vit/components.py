"""BinaryViT components: binary/ternary linear, patch embed, self-attention, MLP, block, norm.

All binary operations use sign-binarization with polynomial surrogate gradient
(no Gumbel, no vanilla STE). Ternary weights use 2-bit {-1, 0, +1} with a
learnable threshold for pruning to zero.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Surrogate gradient for sign()
# -----------------------------------------------------------------------------
class SignPoly(torch.autograd.Function):
    """sign(x) forward, polynomial surrogate backward.

    grad = (1 - |x|^p) for |x| < 1, else 0. p=2 by default.
    """

    @staticmethod
    def forward(ctx, x, p=2.0):
        ctx.save_for_backward(x)
        ctx.p = p
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        p = ctx.p
        grad = F.relu(1.0 - x.abs().pow(p))
        return grad * grad_output, None


def sign_poly(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    return SignPoly.apply(x, p)


# -----------------------------------------------------------------------------
# Binary linear: weights binarized to ±1
# -----------------------------------------------------------------------------
class BinaryLinear(nn.Module):
    """Linear with weights binarized to ±1 via polynomial surrogate."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, p: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # init with kaiming, but small std so initial binarization is not too random
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_b = sign_poly(self.weight, p=self.p)
        else:
            w_b = torch.sign(self.weight)
        return F.linear(x, w_b, bias=self.bias)


# -----------------------------------------------------------------------------
# Ternary linear: weights binarized to {-1, 0, +1}
# -----------------------------------------------------------------------------
class TernaryLinear(nn.Module):
    """Linear with weights ternarized to {-1, 0, +1}.

    Uses a learnable per-tensor threshold: |w| < threshold -> 0,
    else sign(w). Threshold is initialized to 0.5 * mean(|w|) (TWN-style).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 threshold_init: float = 0.05, p: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        # learnable threshold for pruning to zero
        self.threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self._bit_width = 2  # ternary = log2(3) ~ 1.58 bits, round to 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # ternarize with polynomial surrogate
            t = torch.sigmoid(self.threshold)  # in (0, 1)
            w = self.weight
            # ternarize: 0 if |w| < t, else sign(w)
            mask = (w.abs() >= t).float()
            w_t = sign_poly(w, p=self.p) * mask
            # STE for the mask (so gradient flows through threshold)
            mask_ste = mask + (mask - mask.detach())
            w_t = sign_poly(w, p=self.p) * mask_ste
        else:
            t = torch.sigmoid(self.threshold)
            mask = (self.weight.abs() >= t).float()
            w_t = torch.sign(self.weight) * mask
        return F.linear(x, w_t, bias=self.bias)


# -----------------------------------------------------------------------------
# Binary patch embedding: split image into patches, project each to D-dim binary token
# -----------------------------------------------------------------------------
class BinaryPatchEmbed(nn.Module):
    """Patch embedding: image -> (N, num_patches, D) binary tokens.

    Uses a strided unfold + binary linear projection.
    """

    def __init__(self, img_size: int = 28, patch_size: int = 4, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        # patch flatten -> binary linear projection
        self.proj = BinaryLinear(in_channels * patch_size * patch_size, embed_dim, bias=False)
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> (N, num_patches, embed_dim)"""
        N, C, H, W = x.shape
        p = self.patch_size
        # unfold: (N, C, num_patches_h, num_patches_w, p, p)
        x = x.unfold(2, p, p).unfold(3, p, p)  # (N, C, nph, npw, p, p)
        x = x.contiguous().view(N, C, -1, p * p)  # (N, C, num_patches, p*p)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, num_patches, C, p*p)
        x = x.view(N, self.num_patches, -1)  # (N, num_patches, C*p*p)
        return self.proj(x)  # (N, num_patches, embed_dim)


# -----------------------------------------------------------------------------
# Binary self-attention
# -----------------------------------------------------------------------------
class BinarySelfAttention(nn.Module):
    """Multi-head self-attention with binary Q, K, V projections.

    The attention scores use a sparse top-k mask instead of softmax
    (more stable with binary QK products which have high variance).
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.0,
                 topk: int = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = BinaryLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = BinaryLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = BinaryLinear(embed_dim, embed_dim, bias=False)
        self.out_proj = BinaryLinear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.topk = topk  # if set, use sparse top-k attention
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L, D) -> (N, L, D)"""
        N, L, D = x.shape
        q = self.q_proj(x).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, L, hd)
        k = self.k_proj(x).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (N, H, L, L)
        if self.topk is not None and self.topk < L:
            # sparse top-k attention: only keep top-k scores per row
            topk = min(self.topk, L)
            topk_vals, topk_idx = scores.topk(topk, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores * mask + (1 - mask) * (-1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (N, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(N, L, D)
        return self.out_proj(out)


# -----------------------------------------------------------------------------
# Binary MLP
# -----------------------------------------------------------------------------
class BinaryMLP(nn.Module):
    """FFN: D -> 4D -> D with binary linear + GELU."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        self.fc1 = BinaryLinear(embed_dim, hidden_dim, bias=False)
        self.fc2 = BinaryLinear(hidden_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------
# Binary-friendly LayerNorm: per-token mean subtraction + scale (no FP running stats)
# -----------------------------------------------------------------------------
class BinaryLayerNorm(nn.Module):
    """Simplified LayerNorm for binary networks: subtract mean, divide by std,
    then apply learnable scale (no bias to keep things minimal).
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.gamma + self.beta


# -----------------------------------------------------------------------------
# ViT block: LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual
# -----------------------------------------------------------------------------
class BinaryViTBlock(nn.Module):
    """Standard ViT block with binary components."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_hidden: int = None,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = BinaryLayerNorm(embed_dim)
        self.attn = BinarySelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = BinaryLayerNorm(embed_dim)
        self.mlp = BinaryMLP(embed_dim, mlp_hidden, dropout)
        self._bit_width = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
