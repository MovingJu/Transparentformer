# Transparentformer

Transformer를 라이브러리(`torch.nn.MultiheadAttention`, `nn.Transformer` 등)에 기대지 않고
**밑바닥부터 직접 구현**하면서 원리를 이해하는 학습 레포.

## 진행 방식

`asciify` 프로젝트와 같은 방식:

1. 이슈가 순서대로 올라와 있음 — 위에서부터 순서대로 진행.
2. 각 이슈에 `## 목표` / `## 할 일` / `## 검증 방법` / `## 참고`가 있음. 체크리스트를 끝내고
   PR을 올리면 리뷰함.
3. PR 본문에 어떤 이슈를 다루는지 적고(`closes #N`), 커밋은 `feat:`/`fix:` 등 컨벤션대로.
4. 브랜치명은 `issue-N` 패턴.

## 로드맵

| # | 내용 |
|---|------|
| [#1](../../issues/1) | 환경 세팅 + Scaled Dot-Product Attention |
| [#2](../../issues/2) | Multi-Head Attention |
| [#3](../../issues/3) | Positional Encoding (sinusoidal) |
| [#4](../../issues/4) | Feed-Forward + Residual + LayerNorm → Encoder Layer 완성 |
| [#5](../../issues/5) | Encoder 전체 조립 (레이어 쌓기 + padding mask) |
| [#6](../../issues/6) | Decoder Layer (causal mask + cross-attention) |
| [#7](../../issues/7) | Toy task로 전체 Seq2Seq Transformer 학습 |
| [#8](../../issues/8) | Decoder-only 미니 GPT로 변형 + character-level 언어모델 |
| [#9](../../issues/9) | Attention weight 시각화 |
| [#10](../../issues/10) | KV cache로 생성 속도 개선 |
| [#11](../../issues/11) | 실제 데이터로 테스트 + 버그 수정 |
| [#12](../../issues/12) | (도전) RoPE / ALiBi 등 최신 positional encoding |
| [#13](../../issues/13) | (도전) 학습된 모델 데모 공유 |

## 개발 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`tests/test_scaffold.py`가 통과하면 환경 세팅은 끝난 것.

## 원칙

- **autograd는 씀** (PyTorch), 하지만 attention/positional encoding/encoder-decoder 구조 같은
  "모듈"은 전부 직접 구현. `nn.MultiheadAttention`, `nn.Transformer*` 클래스는 참고용으로만
  보고 실제 구현엔 쓰지 않기.
- 각 이슈는 **검증 가능한 산출물**이 있어야 함 (shape가 맞는지, loss가 실제로 줄어드는지,
  손으로 계산한 값과 일치하는지 등) — "일단 에러 없이 돌아간다"가 아니라 "맞게 짰다"를
  확인할 수 있는 테스트를 같이 짜는 게 핵심.
