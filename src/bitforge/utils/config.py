"""Config loader: yaml + argparse overrides via dot-paths.

The Config object stores everything in a single `raw` dict; the `data`,
`model`, `train`, etc. properties read from `raw` on access so that
`cfg.set('model.name', X)` is reflected in `cfg.model['name']`.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, Optional

import yaml


class Config:
    """A nested dict-like config with dot-path access.

    All data is stored in `self.raw`. Section accessors (`cfg.model`,
    `cfg.data`, ...) return live views into `raw`, so mutations via
    `cfg.set(...)` are immediately visible.
    """

    __slots__ = ("raw",)

    SECTIONS = ("data", "model", "train", "eval", "runtime")

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        self.raw: Dict[str, Any] = raw if raw is not None else {}

        # ensure all sections exist
        for s in self.SECTIONS:
            self.raw.setdefault(s, {})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(raw=copy.deepcopy(d))

    # ----- section accessors (live views) -----
    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.setdefault("data", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.setdefault("model", {})

    @property
    def train(self) -> Dict[str, Any]:
        return self.raw.setdefault("train", {})

    @property
    def eval(self) -> Dict[str, Any]:
        return self.raw.setdefault("eval", {})

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.raw.setdefault("runtime", {})

    # ----- dot-path access -----
    def get(self, dotpath: str, default: Any = None) -> Any:
        """Access nested keys via 'section.key' or 'section.sub.key'."""
        node: Any = self.raw
        for part in dotpath.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotpath: str, value: Any) -> None:
        """Set a nested key via 'section.key' or 'section.sub.key'."""
        parts = dotpath.split(".")
        node = self.raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def __repr__(self) -> str:
        return f"Config({self.raw!r})"


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config file and optionally apply dot-path overrides.

    `overrides` example: {"train.batch_size": 64, "model.width": 1.5}
    """
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    cfg = Config.from_dict(d)
    if overrides:
        for k, v in overrides.items():
            cfg.set(k, v)
    return cfg
