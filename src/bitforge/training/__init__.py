"""Training infrastructure: surrogates, optimizers, schedulers, BitNorm, trainer."""
from bitforge.training.surrogates import (
    SignSTE,
    PolytanhSurrogate,
    ATanSurrogate,
    SwishSurrogate,
    HardTanhSurrogate,
    SurrogateBin,
    SURROGATES,
)
from bitforge.training.bitnorm import BitNorm, PopCountNorm
from bitforge.training.trainer import Trainer
from bitforge.training.schedulers import build_scheduler

__all__ = [
    "SignSTE", "PolytanhSurrogate", "ATanSurrogate", "SwishSurrogate",
    "HardTanhSurrogate", "SurrogateBin", "SURROGATES",
    "BitNorm", "PopCountNorm",
    "Trainer", "build_scheduler",
]
