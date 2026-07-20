# Phase 2 — Transformer 블록 조립

## #10 Sinusoidal Positional Encoding

- **왜 위치 정보가 필요한가**: attention의 `QKᵀ`는 순서를 바꿔도 값이 같다(순열 불변). "나는
  너를 사랑해"와 "너는 나를 사랑해"를 구분 못 한다. 그래서 각 위치에 고유한 신호를 더한다.
- **sinusoidal 공식**: 위치 pos, 차원 i에 대해
  `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(...)`.
  차원마다 파장이 다른 sin/cos을 쌓는다 — 낮은 차원은 빠르게 진동(가까운 위치 구분), 높은
  차원은 천천히(먼 위치 구분). 여러 주파수를 겹쳐 "위치의 지문"을 만든다.
- **왜 sin/cos인가**: 상대 위치 `PE[pos+k]`가 `PE[pos]`의 선형변환으로 표현돼서, 모델이
  "상대적 거리"를 학습하기 쉽다. 또 학습 파라미터가 없어(고정) 임의 길이로 확장 가능.
- 학습형(learned) PE(`nn.Embedding`으로 위치를 학습)도 있다 — #24의 RoPE/ALiBi와 함께 비교.

## #11 LayerNorm

- **정규화가 왜 필요한가**: 레이어를 깊이 쌓으면 각 층의 출력 분포가 계속 흔들려 학습이
  불안정해진다. 정규화로 각 토큰 벡터를 mean 0, var 1로 맞춰 안정시킨다.
- **LayerNorm vs BatchNorm**: BatchNorm은 "배치 축"으로 통계를 낸다 → 배치 크기·시퀀스 길이에
  민감. LayerNorm은 **각 토큰의 feature 축(d_model)** 하나만으로 통계를 낸다 → 배치/길이와
  무관. 그래서 NLP·트랜스포머는 LayerNorm.
- **공식**: `y = (x - mean) / √(var + ε) * γ + β`. mean/var는 마지막 축 기준. `γ,β`는 학습되는
  scale/shift 파라미터. `ε`는 0 나눗셈 방지.
- **pre-norm vs post-norm**: 정규화를 서브레이어 "앞"에 두느냐(pre) "뒤"(post, 원논문)에 두느냐가
  학습 안정성을 크게 좌우한다. pre-norm이 깊은 모델에서 더 안정적이라 요즘 표준.

## #12 (🔧 C) LayerNorm forward+backward

- **`autograd.Function`이 뭔가**: forward와 backward를 직접 정의해 PyTorch autograd 그래프에
  끼워넣는 방법. `forward(ctx, ...)`에서 필요한 값을 `ctx.save_for_backward`로 저장하고,
  `backward(ctx, grad_output)`에서 입력에 대한 gradient를 반환한다.
- **backward가 받는 것/주는 것**: `grad_output`(출력에 대한 loss의 미분)을 받아, 연쇄법칙으로
  **입력과 파라미터(γ,β)에 대한 미분**을 계산해 돌려준다. "출력쪽 gradient → 입력쪽 gradient"로
  거꾸로 흘리는 게 backward의 본질.
- **LayerNorm의 미분**: `y=(x-μ)/σ·γ+β` 를 x에 대해 미분하면 μ와 σ가 x에 의존하므로 항이
  여러 개 나온다(정규화 특유의 "평균/분산 경로"). γ,β의 미분은 간단
  (`∂L/∂γ = Σ grad·x̂`, `∂L/∂β = Σ grad`).
- **`gradcheck`**: 내 backward가 맞는지 PyTorch가 수치미분(작은 ε 흔들어 차분)으로 대조해주는
  도구. 손미분 검증의 표준. `double` 정밀도에서 해야 통과가 쉽다.

## #13 FFN + Residual + Norm → Encoder Layer

- **Encoder Layer의 두 서브레이어**: ① Multi-Head **Self**-Attention, ② **Feed-Forward
  Network(FFN)**. 각 서브레이어를 residual + LayerNorm이 감싼다.
- **FFN(position-wise)**: `Linear(d→d_ff) → 비선형(ReLU/GELU) → Linear(d_ff→d)`. 각 토큰에
  **독립적으로** 적용되는 작은 MLP. attention이 "토큰 간 정보 교환"이라면 FFN은 "토큰별
  정보 가공". `d_ff`는 보통 `4*d_model`.
- **residual connection**: `x + Sublayer(x)`. gradient가 깊은 층을 지나도 죽지 않게(항등 경로)
  해준다 — 이게 없으면 깊은 트랜스포머는 학습이 거의 안 된다.
- **pre-norm 구조**(권장): `x + Sublayer(LayerNorm(x))`.

## #14 Encoder 스택

- **임베딩 → PE → N개 레이어**의 전체 파이프라인: 토큰 id를 `nn.Embedding`으로 벡터화 →
  positional encoding 더하기 → EncoderLayer를 N번 통과.
- **왜 그냥 쌓나**: 각 레이어가 표현을 조금씩 정제(refine)한다. 아래층은 국소적/문법적 패턴,
  위층은 추상적/의미적 패턴을 잡는 경향(#21 시각화로 관찰 가능).
- **mask 전파**: padding mask는 모든 레이어의 self-attention에 동일하게 전달돼야 한다. 한 곳만
  빠뜨려도 pad 토큰이 정보를 오염시킨다.

## #15 Decoder Layer + 스택

- **Decoder Layer의 세 서브레이어**: ① **Masked self-attention**(지금까지 생성한 토큰만, causal
  mask), ② **Cross-attention**(Query는 디코더, **Key/Value는 인코더 출력** — 인코더-디코더를
  잇는 다리), ③ FFN.
- **왜 self는 masked, cross는 안 masked인가**: self는 미래 토큰을 막아야 하지만(생성은 순차),
  cross에서 보는 인코더 출력은 "이미 다 아는 입력"이라 전부 봐도 된다(단 padding mask는 적용).
- **cross-attention의 shape**: Query 길이 = 타겟 길이 T_tgt, Key/Value 길이 = 소스 길이 T_src.
  attention 행렬이 `(T_tgt, T_src)`로 **정사각형이 아닐 수 있다** — self-attention과 다른 점.

## #16 전체 조립 + 출력 head

- **전체 데이터 흐름**: `src_ids → Encoder → memory` / `tgt_ids + memory → Decoder → hidden →
  Linear(d_model→vocab) → logits`. logits에 softmax하면 다음 토큰 확률.
- **weight tying**: 입력 임베딩과 출력 projection의 가중치를 공유하면 파라미터가 줄고 성능이
  좋아지는 경우가 많다.
- **teacher forcing**(예고): 학습 때 디코더 입력으로 "정답 시퀀스를 한 칸 민 것"을 넣는다 — 본격
  적용은 #17에서.
