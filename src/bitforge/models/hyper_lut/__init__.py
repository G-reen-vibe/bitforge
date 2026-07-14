"""Hyper-LUT Net: hybrid HDC encoding + differentiable LUT layers.

Architecture (skeleton — design iteration will happen in a later milestone):
    1. Random binary projection encoder:    input (C, H, W) -> hypervector (D=4096, ±1)
       (random Gaussian projection, sign-binarized, fixed — no learning)
    2. LUT blocks: each block consists of:
         - take k coordinates of the input HV (with positional permutation)
         - apply a differentiable k-input LUT (2^k entries, Gumbel-softmax)
         - bundle outputs (majority vote) to produce next HV
    3. Permutation between blocks (HDC-style orthogonal reordering)
    4. Readout: bundled final HV -> binary linear classifier (XNOR-popcount)

The entire model is binary at inference: every parameter is ±1 (LUT tables
collapse to binary selections post-annealing; the binary linear classifier
is the only "wide" binary matmul).

Training uses Gumbel-softmax relaxation for the LUT tables (no STE anywhere).
"""
from bitforge.models.hyper_lut.encoder import HDCEncoder
from bitforge.models.hyper_lut.lut_layer import DifferentiableLUT, LUTBlock
from bitforge.models.hyper_lut.bundling import majority_bundle
from bitforge.models.hyper_lut.model import HyperLUTNet

__all__ = ["HDCEncoder", "DifferentiableLUT", "LUTBlock", "majority_bundle", "HyperLUTNet"]
