"""이 테스트 하나만 통과하면 환경 세팅은 끝난 것 — 패키지가 import되고 torch가 잡히는지만 확인."""

import torch

import transparentformer


def test_package_imports():
    assert transparentformer is not None


def test_torch_available():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0
