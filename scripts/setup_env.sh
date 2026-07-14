#!/bin/bash
# Setup script: install dependencies and verify the install
set -e
cd "$(dirname "$0")/.."

echo "=== Installing PyTorch (CPU) ==="
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing remaining requirements ==="
pip install --no-cache-dir -r requirements.txt

echo "=== Installing bitforge (editable) ==="
pip install -e .

echo "=== Running smoke tests ==="
python -m pytest tests/test_models.py tests/test_utils.py -v

echo "=== Done ==="
