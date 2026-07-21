import torch, pytest
from types import ModuleType
from transparentformer.cext import load_kernels
from benchmark_fixture import BenchmarkFixture


@pytest.fixture
def two_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(64, 64, dtype=torch.float32),
        torch.randn(64, 64, dtype=torch.float32),
    )


@pytest.fixture
def get_kernel() -> ModuleType:
    return load_kernels()


@pytest.mark.benchmark(group="matmul")
def test_matmul_c(
    benchmark: BenchmarkFixture,
    get_kernel: ModuleType,
    two_tensors: tuple[torch.Tensor, torch.Tensor],
):
    A, B = two_tensors
    benchmark(get_kernel.matmul, A, B)


@pytest.mark.benchmark(group="matmul")
def test_matmul_torch(
    benchmark: BenchmarkFixture, two_tensors: tuple[torch.Tensor, torch.Tensor]
):
    def matmul(A: torch.Tensor, B: torch.Tensor):
        return A @ B

    A, B = two_tensors
    benchmark(matmul, A, B)
