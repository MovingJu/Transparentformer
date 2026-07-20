# Phase 0 — 기반 다지기

## #1 환경 세팅

- **왜 `src/` 레이아웃인가**: 패키지 코드를 `src/transparentformer/`에 두고 `pip install -e`로
  설치하면, 테스트가 "설치된 패키지"를 import하게 되어 실제 배포 상황과 같아진다. 루트에 그냥
  두면 "우연히 같은 폴더라서 import되는" 함정에 빠지기 쉽다.
- **editable install (`-e`)**: 코드를 고칠 때마다 재설치할 필요 없이 소스를 바로 반영해주는
  설치 방식. 학습 레포에 딱 맞음.
- **가상환경(venv)**: 시스템 파이썬을 더럽히지 않고 이 프로젝트만의 의존성을 격리.

## #2 텐서·einsum 워밍업

- **차원(shape) 읽는 법**: `(batch, seq_len, d_model)` 같은 텐서에서 각 축이 무슨 의미인지
  항상 주석으로 달아두는 습관. 트랜스포머 버그의 90%는 축을 헷갈려서 생긴다.
- **broadcasting**: shape가 다른 두 텐서를 곱/합할 때 PyTorch가 자동으로 축을 맞춰주는 규칙.
  예: `(B, T, D) + (D,)` 는 마지막 축이 맞으므로 자동 확장된다. 규칙을 정확히 알아야
  "왜 이게 되고 저건 안 되는지"가 보인다.
- **`einsum`**: 아인슈타인 표기법. `torch.einsum("bik,bkj->bij", A, B)` 는 "배치 b마다
  A(i×k)와 B(k×j)를 곱해 (i×j)를 만들라"는 뜻. 반복되는 인덱스(k)는 곱해서 합(sum)하고,
  출력에 없는 인덱스는 사라진다. attention의 `QKᵀ`, `weights·V` 가 전부 einsum 한 줄로 표현된다.

## #3 (🔧 C) matmul 커널 + cpp_extension

- **PyTorch C++ extension이 뭔가**: PyTorch는 텐서를 C/C++에서 직접 다루는 API(`torch::Tensor`,
  또는 C 수준에서 `data_ptr<float>()`로 raw 포인터)를 제공한다. `cpp_extension`은 `.cpp` 파일을
  그때그때 컴파일해서(JIT) 파이썬 모듈로 로드해준다. 내부적으로 pybind11을 씀 — 순수
  Python/C API보다 보일러플레이트가 훨씬 적다.
- **`load()` (JIT) vs `setup.py build`**: 처음엔 `torch.utils.cpp_extension.load(...)`로 즉석
  빌드가 편함. 셋업 파일 없이 함수 하나 호출로 컴파일+로드.
- **메모리 레이아웃**: 텐서는 메모리상 1차원으로 쭉 늘어서 있고(row-major), `A[i][j]`는
  `data[i * ncols + j]`로 접근한다. C에서 이 인덱싱을 직접 하는 감각이 이후 모든 C 이슈의 기본.
- **naive matmul의 3중 루프**와 왜 느린가(캐시 미스), 블록 단위로 나누면(타일링) 왜 빨라지는가.
- **검증 패턴**: 부동소수 누적 순서가 달라 완전 일치는 안 되므로 `torch.allclose(..., atol=1e-4)`
  로 대조한다 — 이 패턴이 앞으로 모든 C 이슈의 표준 검증 방식.

## #4 토이 MLP 학습 루프

- **학습 루프의 5단계**: ① forward로 예측 → ② loss 계산 → ③ `loss.backward()`로 gradient →
  ④ `optimizer.step()`으로 파라미터 갱신 → ⑤ `optimizer.zero_grad()`로 gradient 초기화. 이
  순서와 특히 **zero_grad를 빼먹으면 gradient가 누적되는** 함정을 몸으로 익힌다.
- **autograd가 해주는 것**: forward만 짜면 PyTorch가 계산 그래프를 기록해두었다가 backward를
  자동으로 해준다. 이 레포는 이 autograd만 빌려 쓰고 "모듈 구조"는 직접 짠다.
- **왜 XOR인가**: 선형 분리가 불가능한 최소 예제 — 은닉층 없는 선형 모델은 절대 못 풀고,
  은닉층+비선형(ReLU/tanh)이 있어야 풀린다. "비선형성이 왜 필요한가"를 눈으로 확인.
- **loss가 내려간다 = 학습이 된다**의 감각. 앞으로 모든 학습 이슈의 검증 기준.
