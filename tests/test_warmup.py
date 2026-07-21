"""Tests for transparentformer.warmup module"""

import torch
import transparentformer


def test_warmup():
    # basic tensor test.
    a = torch.randn(4, 3, 5)
    b = torch.randn(4, 5, 2)
    out = transparentformer.warmup.batch_mat_mul(a, b)
    assert torch.allclose(out, torch.bmm(a, b))

    # atttention head test.
    batch_size, seq_len, d_model, _num_heads = (2, 5, 8, 2)
    x_t = torch.randn(batch_size, seq_len, d_model)
    Q = x_t @ torch.randn(d_model, d_model)
    K = x_t @ torch.randn(d_model, d_model)
    QK_t = transparentformer.warmup.batch_mat_mul(Q, K.transpose(1, 2))
    assert QK_t.shape == torch.Size([2, 5, 5])
