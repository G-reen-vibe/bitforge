"""Logging + timing utilities."""
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def get_logger(name: str = "bitforge", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


@dataclass
class StageTimer:
    """Accumulate wall-clock time across named stages."""
    times: Dict[str, float] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self._starts[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self._starts:
            return 0.0
        dt = time.time() - self._starts.pop(name)
        self.times[name] = self.times.get(name, 0.0) + dt
        return dt

    @contextmanager
    def section(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def summary(self) -> Dict[str, float]:
        return dict(self.times)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
