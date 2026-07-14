"""BitForge: A 1-bit machine learning framework built from scratch.

Primary research direction: Hyper-LUT Net — a hybrid of Hyperdimensional
Computing (HDC) input encoding and differentiable LUT layers trained with
Gumbel-softmax relaxation, with no straight-through estimator (STE) anywhere.

Baselines reproduced for fair comparison:
    - FP-ResNet (full-precision upper bound)
    - XNOR-Net (classic BNN, 2016)
    - Bi-Real Net (2018)
    - IR-Net (2020)
    - ReActNet (2020, SOTA binary CNN)

Supported benchmarks: MNIST, CIFAR-10, CIFAR-100.
"""

__version__ = "0.1.0"
