## 문제 정의
- NSMC 감성분석
- 입력: 리뷰
- 출력: 긍정/부정

## 모델

Input
↓
Embedding
↓
LSTM
↓
Last Hidden State
↓
Linear
↓
Softmax

Input Shape : (batch, seq_len)
Embedding : (batch, seq_len, embed_dim)
...

## 실험 계획