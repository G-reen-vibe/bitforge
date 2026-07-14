"""Models: baselines + Hyper-LUT Net."""
import inspect

from bitforge.models.baselines import (
    FPResNet,
    XNORNet,
    BiRealNet,
    IRNet,
    ReActNet,
    BASELINE_REGISTRY,
)
from bitforge.models.hyper_lut import HyperLUTNet

__all__ = [
    "FPResNet", "XNORNet", "BiRealNet", "IRNet", "ReActNet",
    "BASELINE_REGISTRY", "HyperLUTNet",
    "build_model",
]


from bitforge.utils.config import Config


def _filter_kwargs(cls, kwargs):
    """Filter kwargs to only those accepted by cls.__init__."""
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def build_model(cfg: Config):
    """Build a model from config: cfg.model.name + cfg.model.<kwargs>.

    kwargs not accepted by the target model's __init__ are silently filtered,
    so a config file with Hyper-LUT-specific keys (hv_dim, k, ...) can be
    safely used to build a baseline (FPResNet, ReActNet, ...).
    """
    name = cfg.model.get("name", "hyper_lut").lower()
    raw_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    if name in BASELINE_REGISTRY:
        cls = BASELINE_REGISTRY[name]
        return cls(**_filter_kwargs(cls, raw_kwargs))
    if name in ("hyper_lut", "hyperlut"):
        return HyperLUTNet(**_filter_kwargs(HyperLUTNet, raw_kwargs))
    raise ValueError(f"Unknown model: {name}")
