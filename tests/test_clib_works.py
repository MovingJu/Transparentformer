"""Checks if c-library works and doc-strings."""

import torch
from transparentformer.cext import load_kernels

def test_softmax():
    k = load_kernels()
    a = torch.rand(2, 3, dtype=torch.float32)
    out = k.softmax(a)
    print(out)
    assert out.shape == (2, 3)