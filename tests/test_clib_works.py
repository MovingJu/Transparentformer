"""Checks if c-library works and doc-strings."""

import torch
from transparentformer.cext import load_kernels

def test_matmul():
    k = load_kernels()
    A = torch.randn(64, 64, dtype=torch.float32)
    B = torch.randn(64, 64, dtype=torch.float32)
    assert torch.allclose(k.matmul(A, B), A @ B, atol=1e-4)

def test_softmax():
    k = load_kernels()
    a = torch.rand(2, 3, dtype=torch.float32)
    out = k.softmax(a)
    print(out)
    assert out.shape == (2, 3)
