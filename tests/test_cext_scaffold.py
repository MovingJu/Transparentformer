"""C 트랙 스캐폴드 확인용 — kernels/(cabin) + csrc/binding.cpp가 실제로 빌드·로드되는지만 본다.
개별 커널의 정확성 테스트는 각 이슈(#3, #6, #8, #12, #20)에서 추가한다.
"""

import torch

from transparentformer.cext import load_kernels


def test_kernels_load_and_callable():
    k = load_kernels()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    out = k.matmul(a, b)
    assert out.shape == (2, 4)
