# Phase 4 — 들여다보고 빠르게 만들기

## #20 (🔧 C) SDPA backward — C 트랙 최종 보스

- **왜 어려운가**: attention은 matmul → (scale) → softmax → matmul의 합성이라, backward도
  각 단계를 역순으로 연쇄해야 한다. 특히 **softmax의 Jacobian**이 까다롭다.
- **softmax backward**: `y=softmax(x)`일 때 `∂L/∂x = y ⊙ (g - (g·y))` (g=∂L/∂y). 즉 각 행에서
  "gradient에서 그 행의 가중평균을 뺀" 형태.
- **연쇄 정리**: `O = P·V` (P=softmax결과) → `dV = Pᵀ·dO`, `dP = dO·Vᵀ` → dP를 softmax
  backward로 통과 → `dScores` → `dQ = dScores·K·scale`, `dK = dScoresᵀ·Q·scale`.
- **`save_for_backward`로 뭘 저장하나**: backward에 필요한 forward 중간값(P 또는 Q,K,V와
  softmax 출력)을 forward에서 저장해둬야 한다. 메모리/재계산 트레이드오프의 실물 예.
- **gradcheck의 위력**: 손유도가 한 항이라도 틀리면 gradcheck가 잡아낸다.

## #21 Attention weight 시각화

- **attention weight = 해석 창구**: `(T, T)` 가중치 행렬의 `w[i][j]`는 "토큰 i가 j를 얼마나
  봤나". heatmap으로 그리면 모델의 "시선"이 보인다.
- **무엇이 보이나**: 복사 task면 대각선(자기 위치), 뒤집기면 반대각선, 언어모델이면 직전
  토큰이나 관련 단어에 쏠린 패턴. head마다 다른 패턴(#9의 "head가 서로 다른 관계를 본다"가
  실제로 관찰됨).
- **레이어별 차이**: 아래 레이어는 국소적, 위 레이어는 넓은 문맥을 보는 경향.
- **디버깅 도구로서의 시각화**: 학습이 이상하면 attention이 다 균일(uniform)하거나 한 토큰에만
  붙어있는 등 증상이 그림으로 드러난다.

## #22 KV cache

- **생성의 비효율**: 순차 생성에서 매 스텝 전체 시퀀스를 다시 넣으면, 이미 계산했던 앞
  토큰들의 Key/Value를 매번 다시 계산한다. T 토큰 생성에 O(T²)의 중복.
- **핵심 통찰**: causal attention에서 과거 토큰의 K,V는 **미래에 바뀌지 않는다**. 한 번 계산해
  캐시에 쌓아두고, 새 토큰의 K,V만 추가하면 된다.
- **효과**: 스텝당 비용이 O(T)→O(1) 수준으로 떨어져 긴 생성이 크게 빨라진다.
- **정확성 불변 조건**: 캐시를 써도 **출력은 캐시 없는 버전과 완전히 동일**해야 한다. 빨라지되
  결과가 바뀌면 버그.

## #23 실데이터 테스트

- **엣지 케이스가 버그를 드러낸다**: 아주 짧은/긴 시퀀스, 한 배치에 길이 편차가 큰 경우,
  `max_len` 초과, 빈 입력 등에서 shape·mask·PE 버그가 튀어나온다.
- **일반화 vs 암기**: 학습에 없던 입력에서도 그럴듯한지 — 과적합만 된 건지 실제 패턴을 배운
  건지.
