# Phase 1 — Attention 코어

## #5 Scaled Dot-Product Attention

`Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V`

- **Q, K, V의 직관**: 각 토큰은 "내가 찾는 것"(Query), "내가 가진 열쇠"(Key), "내가 줄 값"(Value)을
  갖는다. Query와 모든 Key의 내적으로 "얼마나 관련 있나"(점수)를 재고, 그 점수로 Value를
  가중평균한다. 즉 **"관련 있는 토큰의 정보를 더 많이 가져오는" 연산.**
- **`QKᵀ`**: `(T, dₖ)·(dₖ, T) = (T, T)` — 모든 토큰 쌍의 유사도 행렬. `scores[i][j]` = 토큰 i가
  토큰 j를 얼마나 볼지.
- **왜 `√dₖ`로 나누나 (scaling)**: dₖ가 크면 내적 값의 분산이 커져서 softmax가 한 곳으로 확
  쏠리고(포화) gradient가 죽는다. `√dₖ`로 나눠 분산을 1 수준으로 유지 → 학습 안정.
- **softmax**: 점수를 확률(합=1)로 바꾼다. **수치안정성**이 중요 — 큰 값이 들어오면 `exp`가
  overflow하므로, 항상 `x - max(x)`를 먼저 빼고 exp한다(결과는 수학적으로 동일).
- **가중합 `·V`**: `(T, T)·(T, dᵥ) = (T, dᵥ)` — 확률로 Value를 섞어 각 토큰의 새 표현을 만든다.

## #6 (🔧 C) softmax 커널

- **reduction 패턴**: softmax는 마지막 축을 따라 ① max 구하기 → ② `exp(x-max)` → ③ 그 합
  구하기 → ④ 합으로 나누기. 각 단계가 "한 행을 훑어 하나의 값으로 줄이는(reduce)" 루프다.
- **왜 max-subtract가 수치안정성인가 (다시)**: `exp(1000)`은 float에서 `inf`. 하지만
  `exp(1000-1000)=exp(0)=1`. 각 행의 max를 빼면 가장 큰 항이 `exp(0)=1`이 되어 overflow가
  원천 차단된다. 분모/분자에 같은 상수를 곱한 셈이라 결과는 불변.
- **in-place vs 새 버퍼**: 출력을 입력과 같은 버퍼에 쓸지 새로 잡을지 — C에서 메모리 관리를
  직접 결정하는 감각.

## #7 마스킹 (padding / causal)

- **왜 softmax "전에" −∞를 더하나**: 마스크할 위치의 점수에 `-inf`(실제론 아주 큰 음수)를
  더하면, `exp(-inf)=0`이 되어 그 위치의 attention weight가 정확히 0이 된다. softmax 후에
  0을 곱하면 남은 확률들의 합이 1이 안 되므로, 반드시 softmax **전에** 처리해야 한다.
- **padding mask**: 문장마다 길이가 다르니 짧은 건 `<pad>`로 채운다. 그 자리는 정보가 없으므로
  누구도 그걸 보면 안 된다. shape는 보통 `(B, 1, 1, T)`로 만들어 broadcasting.
- **causal(look-ahead) mask**: 언어모델은 다음 단어를 "예측"해야 하는데, 학습 때 미래 단어를
  보면 커닝이다. 그래서 토큰 i는 j≤i만 보게 상삼각을 막는다. `torch.tril`로 하삼각 1 행렬을
  만들어 그 바깥을 −∞. 결과 weight 행렬이 **엄밀히 하삼각**이어야 한다.
- `-inf` 대신 `-1e9` 같은 큰 음수를 쓰면 NaN 위험이 줄어든다(한 행이 전부 마스크되면 `-inf`끼리
  softmax하다 NaN이 남).

## #8 (🔧 C) SDPA forward 전체

- **커널 합성(fusion)의 감각**: 파이썬에선 matmul → softmax → matmul을 세 번의 텐서 연산으로
  했지만, C에선 이걸 한 함수 안에서 중간 버퍼를 직접 관리하며 이어붙인다. (진짜 FlashAttention은
  이 fusion을 극한까지 밀어 중간 `(T,T)` 행렬을 아예 메모리에 안 올린다 — 그 아이디어의 씨앗.)
- **중간 버퍼 관리**: `scores`(T×T)를 담을 임시 메모리를 C에서 `malloc`/`std::vector`로 잡고
  쓰고 해제. 메모리 수명 관리를 직접 하는 연습.
- **scaling 위치**: `QKᵀ` 직후 `1/√d`를 곱하는 걸 잊지 않기 (파이썬 대조로 바로 드러남).

## #9 Multi-Head Attention

- **왜 여러 head인가**: 한 attention은 한 종류의 관계밖에 못 잡는다. head를 h개로 나누면
  "문법적 관계를 보는 head", "지시대명사를 보는 head"처럼 **서로 다른 패턴을 병렬로** 학습할 수
  있다. 각 head의 차원은 `d_model / h`라 총 계산량은 그대로.
- **4단계 흐름**: ① **projection**(`W_q,W_k,W_v`로 Q,K,V 생성) → ② **head 분할**
  (`(B,T,d_model)`→`(B,h,T,d_head)`) → ③ **병렬 attention**(head 축은 배치처럼 취급, #5~#7 코어
  그대로 적용) → ④ **concat + out proj**(`W_o`).
- **self-attention vs cross-attention**: Q,K,V가 같은 입력에서 나오면 self, Q는 디코더·K/V는
  인코더에서 나오면 cross. MHA 코드는 동일하고 입력만 다르게 넣는다.
