"""Reproducibility: seeding for torch, numpy, python."""
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for python, numpy, torch (CPU + CUDA if available).

    For CPU-only PyTorch this fully determines RNG state for given software
    versions. For full determinism in conv ops set `deterministic=True`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # CUBLAS_WORKSPACE_CONFIG needed for deterministic CUDA conv
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded with `seed` (for DataLoaders)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
