# Transparentformer

Transformer를 라이브러리(`torch.nn.MultiheadAttention`, `nn.Transformer` 등)에 기대지 않고
**밑바닥부터 직접 구현**하면서 원리를 이해하는 학습 레포. 이름 그대로 — 안이 다 들여다보이는
(transparent) 트랜스포머를 만드는 게 목표.

## 진행 방식

`asciify` 프로젝트와 같은 방식:

1. 이슈가 순서대로 올라와 있음 — 위에서부터 순서대로 진행.
2. 각 이슈에 `## 목표` / `## 배경지식` / `## 할 일` / `## 검증 방법` / `## 참고`가 있음.
   개념 설명은 이슈에 직접 안 담고 berry docs(`transformer-book/core/`)로 뺐음 — 이슈끼리 개념이
   중복되는 걸 막기 위함. 체크리스트를 끝내고 PR을 올리면 리뷰함.
3. PR 본문에 어떤 이슈를 다루는지 적고(`closes #N`), 커밋은 `feat:`/`fix:` 등 컨벤션대로.
4. 브랜치명은 `issue-N` 패턴.

## 배경지식 문서

개념 설명은 이 레포가 아니라 berry docs(`~/Documents/transformer-book/core/`)에 페이즈별로
정리되어 있음 — 이슈 본문의 `## 배경지식` 줄이 어느 파일인지 알려줌.

## 두 개의 트랙

- 🧠 **본 트랙 (PyTorch)** — 트랜스포머 구조를 개념 단위로 직접 구현.
- 🔧 **C 트랙** — 본 트랙에서 방금 만들고 검증한 연산을, 바로 다음 이슈에서 **C로 다시 짜서**
  연결. 텐서 연산이 메모리에서 실제로 어떻게 도는지까지 파고듦. C 실력과 병행 학습. c-book
  문서를 참고자료로.

### C 트랙 빌드 구조

`torch.utils.cpp_extension` 하나로 다 몰아넣지 않고, 두 층으로 나눴다:

- **`kernels/`** — [Cabin](https://github.com/cabinpkg/cabin) 프로젝트. 진짜 계산 로직(matmul,
  softmax, SDPA, LayerNorm)이 전부 여기 있고, `extern "C"`로 내보냄. `cabin build`/`cabin fmt`/
  `cabin test`로 이 부분만 독립적으로 다룰 수 있음 — **C 트랙 이슈들은 전부 이 안의 함수를
  채우는 것**.
- **`csrc/binding.cpp`** — `torch::Tensor` ↔ raw pointer 변환 + pybind11 등록만 하는 아주 얇은
  glue. `<torch/extension.h>`는 C++ 전용이라 이 경계는 C++ 없이는 못 짜지만, 실제 계산은 전혀
  안 담겨있음.
- **`src/transparentformer/cext.py`** — `load_kernels()` 한 번 호출하면 `cabin build` →
  `torch.utils.cpp_extension.load()`(binding.cpp + kernels가 만든 static lib 링크)까지 알아서
  해줌.

```python
from transparentformer.cext import load_kernels
k = load_kernels()
k.matmul(a, b)  # kernels/src/kernels.c의 matmul_forward를 그대로 호출
```

지금은 `kernels/src/kernels.c`의 모든 함수가 **스텁(0으로 채움)**이라 컴파일은 되지만 결과는
틀림 — 각 이슈(#3, #6, #8, #12, #20)에서 해당 함수를 실제 구현으로 바꾸면 됨.

C 이슈는 항상 대응하는 PyTorch 이슈 **바로 뒤**에 온다 — 방금 만든 PyTorch 출력이 곧 C 구현의
정답지(대조 기준)가 되기 때문. 그리고 난이도가 서서히 오른다:
**matmul(순수 계산) → softmax(reduction) → SDPA(둘을 합침) → LayerNorm(첫 backward) →
SDPA backward(손미분 전체)**. C 트랙이 버거우면 그 이슈만 건너뛰고 본 트랙을 이어가도 됨.

## 설계 원칙

- **autograd는 씀** (PyTorch), 하지만 attention · positional encoding · LayerNorm ·
  encoder/decoder 구조 같은 "모듈"은 전부 직접 구현. `nn.MultiheadAttention`,
  `nn.Transformer*`, `nn.LayerNorm` 같은 완성품 클래스는 **정답 대조용으로만** 쓰고 실제
  구현엔 넣지 않기. (`nn.Linear`, `nn.Embedding` 같은 순수 파라미터 컨테이너는 써도 됨.)
- 각 이슈는 **검증 가능한 산출물**이 있어야 함 — "에러 없이 돌아간다"가 아니라 "맞게 짰다"를
  확인하는 게 핵심. 검증은 세 종류를 섞어 씀:
  1. **손계산 대조** — 작은 예제(2×2 등)를 종이로 계산해서 코드 출력과 일치하는지.
  2. **참조 구현 대조** — 같은 입력에 PyTorch 완성품(`nn.LayerNorm` 등) 또는 앞서 만든 본 트랙
     구현과 `torch.allclose`로 일치하는지. (C 트랙의 주된 검증 방식.)
  3. **학습 신호** — toy task에서 loss가 실제로 내려가고 정확도가 오르는지.
- 한 이슈 = 한 개념. 성급하게 여러 개 묶지 않기. 막히면 앞 이슈의 검증 테스트부터 다시.

## 로드맵

### Phase 0 — 기반 다지기 (트랜스포머 건드리기 전에)

attention과 학습 루프를 **동시에** 처음 보면 뭐가 틀렸는지 원인 분리가 안 됨. 그 둘을 미리 떼둔다.
C 트랙의 빌드 파이프라인도 가장 단순한 커널(matmul)로 여기서 먼저 뚫는다.

| # | | 목표 | 검증 |
|---|---|------|------|
| [#1](../../issues/1) | 🧠 | 환경 세팅 (venv, 패키지, pytest 통과) | `test_scaffold.py` 초록불 |
| [#2](../../issues/2) | 🧠 | 텐서·einsum 워밍업 (broadcasting, batched matmul을 einsum으로) | 직접 짠 batched matmul이 `torch.bmm`과 `allclose` |
| [#3](../../issues/3) | 🔧 | **C로 matmul 커널** — `kernels/`(Cabin) + `csrc/binding.cpp` 빌드 파이프라인 첫 연결 | #2 결과와 tolerance 내 일치 + naive/블록 버전 속도 비교 |
| [#4](../../issues/4) | 🧠 | 토이 MLP 학습 루프 (XOR 또는 소규모 회귀) | loss 감소 + XOR 4케이스 정답 — **학습 루프를 여기서 미리 체득** |

### Phase 1 — Attention 코어

트랜스포머의 심장. 작은 조각으로 쪼개 각각 검증하고, 조각마다 C 버전을 바로 붙인다.

| # | | 목표 | 검증 |
|---|---|------|------|
| [#5](../../issues/5) | 🧠 | Scaled Dot-Product Attention (`scores = QKᵀ/√d` → softmax → 가중합) | 2×2 손계산과 일치 + softmax 수치안정성(큰 값에서 NaN 안 남) |
| [#6](../../issues/6) | 🔧 | **C로 softmax 커널** (max-subtract 수치안정성까지 C로) | #5 안의 softmax 조각과 `allclose`, 큰 값에서도 안정 |
| [#7](../../issues/7) | 🧠 | 마스킹 (padding mask, causal mask를 softmax 전에 −∞로) | 마스크된 위치 weight ≈ 0, causal이 엄밀히 하삼각 |
| [#8](../../issues/8) | 🔧 | **C로 SDPA forward 전체** (#3 matmul + #6 softmax 조각을 합침) | #5 PyTorch 출력과 tolerance 내 일치 + 속도 비교 |
| [#9](../../issues/9) | 🧠 | Multi-Head Attention (projection → head 분할 → 병렬 attention → concat → out proj) | head=1이면 #5와 동일, shape·파라미터 수 검증 |

### Phase 2 — Transformer 블록 조립

Attention을 감싸는 나머지 부품을 붙여 encoder/decoder 완성. LayerNorm에서 C backward를 첫 연습.

| # | | 목표 | 검증 |
|---|---|------|------|
| [#10](../../issues/10) | 🧠 | Sinusoidal Positional Encoding | shape·주기성 확인, heatmap 그려보기 |
| [#11](../../issues/11) | 🧠 | LayerNorm 직접 구현 (`nn.LayerNorm` 안 씀) | `nn.LayerNorm` 출력과 `allclose`, 정규화 후 mean≈0/var≈1 |
| [#12](../../issues/12) | 🔧 | **C로 LayerNorm forward+backward** + `autograd.Function` 첫 연결 | `torch.autograd.gradcheck` 통과 (self-contained reduction 커널) |
| [#13](../../issues/13) | 🧠 | FFN + Residual + Norm → **Encoder Layer** 하나 완성 | 입출력 shape 보존, gradient가 끝까지 흐름 |
| [#14](../../issues/14) | 🧠 | Encoder 스택 (레이어 N개 + padding mask 전파 + 임베딩) | 레이어 수만큼 파라미터 증가, 마스크가 위 레이어까지 전달됨 |
| [#15](../../issues/15) | 🧠 | Decoder Layer (masked self-attn + cross-attn) + Decoder 스택 | causal + cross-attention 동시 동작, cross가 encoder 출력을 봄 |
| [#16](../../issues/16) | 🧠 | 전체 Encoder–Decoder 조립 + 출력 head | 임의 입력이 `(batch, tgt_len, vocab)` 로짓으로 나옴 |

### Phase 3 — 실제로 학습시키기

여기서 처음으로 "진짜 도는" 모델이 된다.

| # | | 목표 | 검증 |
|---|---|------|------|
| [#17](../../issues/17) | 🧠 | Toy seq2seq (문자열 복사/뒤집기/정렬) 전체 학습 | loss 감소 + toy task 정확도 ≈ 100% |
| [#18](../../issues/18) | 🧠 | Decoder-only 미니 GPT + character-level 언어모델 | 작은 코퍼스에 overfit 되고 그럴듯한 텍스트 생성 |
| [#19](../../issues/19) | 🧠 | 생성 디코딩 (greedy / temperature / top-k) | 같은 seed에서 결정론적 greedy, 온도 올리면 다양성 증가 |

### Phase 4 — 들여다보고 빠르게 만들기

이제 학습을 이해했으니, C 트랙의 최종 보스(attention backward 손미분 전체)에 도전.

| # | | 목표 | 검증 |
|---|---|------|------|
| [#20](../../issues/20) | 🔧 | **C로 SDPA backward** 손미분 전체 + `autograd.Function` | `gradcheck` 통과 + 실제 학습에 투입해 loss 동일하게 감소 |
| [#21](../../issues/21) | 🧠 | Attention weight 시각화 (heatmap) | 학습된 모델에서 의미 있는 정렬 패턴이 눈에 보임 |
| [#22](../../issues/22) | 🧠 | KV cache로 생성 속도 개선 | 캐시 유무 출력이 동일 + 토큰당 시간 단축 측정 |
| [#23](../../issues/23) | 🧠 | 실제 데이터로 테스트 + 버그 수정 | 긴 시퀀스/특수 입력에서 안 터지고 결과 타당 |

### Phase 5 — 도전 과제

| # | | 목표 | 검증 |
|---|---|------|------|
| [#24](../../issues/24) | 🧠 | (도전) RoPE / ALiBi 등 최신 positional encoding으로 교체 | sinusoidal 버전과 성능 비교, 상대위치 성질 확인 |
| [#25](../../issues/25) | 🧠 | (도전) 학습된 모델 데모 공유 | 브라우저/CLI에서 생성 데모 동작 |

## 개발 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
pyright   # 정적 타입 체크 (strict 모드)
```

`tests/test_scaffold.py`가 통과하면 환경 세팅은 끝난 것 (= 이슈 #1 완료).
C 트랙(🔧)은 컴파일러(gcc/clang)와 [Cabin](https://github.com/cabinpkg/cabin) 설치가 필요 —
`tests/test_cext_scaffold.py`가 통과하면 C 트랙 빌드 파이프라인도 준비 끝난 것.

> **알려진 함정 (tmpfs 환경)**: `/tmp`가 tmpfs이고 용량이 작은 환경(예: 이 프로젝트를 만든
> Raspberry Pi)에서는 컴파일러가 중간 파일을 쓰다 `No space left on device`로 실패할 수 있다.
> `TMPDIR`을 디스크 백업 경로로 지정하고 실행하면 해결됨:
> `TMPDIR=~/some-disk-path pytest`

### 타입 체크 — pyright (strict)

`pyright`를 `strict` 모드로 설정해뒀다(`pyproject.toml`의 `[tool.pyright]`). mypy도 검토했는데,
같은 코드에 대해 pyright strict가 훨씬 더 엄격하게 잡아냄(Unknown 타입 전파, 어노테이션 누락 등
mypy 기본/strict 모두보다 많은 에러를 검출 — 실제 비교 테스트로 확인). 새 모듈을 추가할 때마다
`pyright`가 깨끗하게 통과하는지 확인하는 걸 습관으로.

## 왜 이 순서인가

- **Phase 0을 먼저** 두는 이유: attention과 학습 루프를 동시에 배우면 원인 분리가 안 됨. MLP로
  학습 루프를 먼저 익혀두면 Phase 3에서 트랜스포머만 새 변수로 남음.
- **Attention을 #5·#7·#9로 쪼갠** 이유: 마스킹은 attention 버그 최대 단골이라 별도 이슈로 떼서
  확실히 검증하고 넘어감.
- **LayerNorm을 직접(#11)** 구현하는 이유: pre-norm/post-norm 위치가 학습 안정성을 크게 좌우하는데
  완성품을 쓰면 그 감을 못 얻음.
- **C 트랙을 사이사이 끼운(#3·#6·#8·#12·#20)** 이유: 각 C 구현은 바로 앞에서 만든 PyTorch 출력을
  정답지로 삼아 대조함 — 그래서 대응하는 본 트랙 이슈 직후에 배치. 난이도도
  matmul→softmax→SDPA→LayerNorm(첫 backward)→SDPA backward로 완만하게 오름. 맨 뒤에 몰아두지
  않아, "지금 배운 걸 바로 C로" 흐름이 유지됨.
