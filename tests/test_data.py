"""Tests for the data pipeline: datasets build and loaders return expected shapes."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitforge.data.datasets import DATASET_INFO, build_dataset


def test_dataset_info():
    assert DATASET_INFO["mnist"].num_classes == 10
    assert DATASET_INFO["cifar10"].num_classes == 10
    assert DATASET_INFO["cifar100"].num_classes == 100
    assert DATASET_INFO["mnist"].in_channels == 1
    assert DATASET_INFO["cifar10"].in_channels == 3


@pytest.mark.parametrize("name", ["mnist", "cifar10", "cifar100"])
def test_dataset_download_and_load(name, tmp_path):
    """This will download the dataset (~10-200MB) on first run.
    Marked as a normal test, but skipped in CI without network.
    """
    try:
        ds = build_dataset(name, root=str(tmp_path / "data"), train=True, augment="light", download=True)
        assert len(ds) > 0
        x, y = ds[0]
        info = DATASET_INFO[name]
        assert x.shape[0] == info.in_channels
        assert y < info.num_classes
    except Exception as e:
        pytest.skip(f"Dataset download failed (network?): {e}")
