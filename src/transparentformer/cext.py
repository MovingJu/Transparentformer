"""C 트랙 확장 로더.

kernels/(cabin 프로젝트)를 먼저 `cabin build`로 빌드해서 static lib를 만들고,
csrc/binding.cpp(얇은 pybind11 glue)와 링크해서 파이썬 모듈로 로드한다.

사용법:
    from transparentformer.cext import load_kernels
    k = load_kernels()
    k.matmul(a, b)
"""

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from torch.utils.cpp_extension import load  # pyright: ignore[reportUnknownVariableType] -- torch 자체 타입 스텁이 불완전

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO_ROOT / "kernels"
BINDING_SRC = REPO_ROOT / "csrc" / "binding.cpp"


def _find_static_lib() -> Path:
    # cabin 빌드 산출물 위치: kernels/build/<profile>/packages/kernels/libkernels.a
    matches = list(KERNELS_DIR.glob("build/*/packages/kernels/libkernels.a"))
    if not matches:
        raise FileNotFoundError(
            "libkernels.a를 못 찾음 — 먼저 `cabin build`가 실행됐는지 확인 "
            f"(찾은 경로: {KERNELS_DIR}/build/*/packages/kernels/)"
        )
    return matches[0]


@lru_cache(maxsize=1)
def load_kernels() -> ModuleType:
    """kernels/를 cabin으로 빌드하고, C 트랙 파이썬 확장 모듈을 로드해서 리턴한다."""
    subprocess.run(["cabin", "build"], cwd=KERNELS_DIR, check=True)
    lib_path = _find_static_lib()

    # torch.utils.cpp_extension.load()는 `ModuleType | str`을 리턴하도록 타입이 잡혀있는데,
    # str이 되는 건 is_standalone=True(실행파일 빌드) 때뿐 — 여긴 기본값(False)만 쓰므로
    # 항상 ModuleType이 나온다는 걸 알고 있어서 assert로 pyright에 알려줌.
    result = load(
        name="transparentformer_kernels",
        sources=[str(BINDING_SRC)],
        extra_include_paths=[str(KERNELS_DIR / "include")],
        extra_ldflags=[str(lib_path)],
        verbose="-v" in sys.argv,
    )
    assert isinstance(result, ModuleType)
    return result
