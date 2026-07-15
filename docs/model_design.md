# Model Design

## 문제 정의
- NSMC 감성분석
- 입력: 리뷰
- 출력: 긍정/부정


## 모델
### 구조
Input<br>
↓<br>
Embedding<br>
↓<br>
Sequence Model1<br>
↓<br>
Sequence Model2<br>
↓<br>
Linear (Affine, input: Last Hidden State)<br>
↓<br>
BCEWithLogitsLoss (Sigmoid, Binary Cross Entropy)

---

### 파라미터

참고: 『파이썬 딥러닝 머신러닝 입문』

| Parameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 (initial) |
| Sequence length | preprocessing 결과 |
| Vocabulary size | preprocessing 결과 |
| Embedding dimension | 128 (initial) |
| Hidden size | 64 (initial) |

**Input**

```
Shape: (batch_size, seq_len)
```

**Embedding**

```
Parameters : (vocab_size, embedding_dim)

Output     : (batch_size, seq_len, embedding_dim)
```

**Sequence Model 1**

```
Parameters : (input_size=embedding_dim, hidden_size)

Output     : (batch_size, seq_len, hidden_size)
```

**Sequence Model 2**

```
Parameters : (input_size=hidden_size, hidden_size)

Output     : (batch_size, seq_len, hidden_size)
```

**Linear**

```
Parameters : (hidden_size, 1)

Output     : (batch_size, 1)
```


## 실험 계획

### 1. 전처리 비교

고정 조건
- 모델: LSTM
- 하이퍼파라미터: 초기값 사용 (필요시 튜닝된 값 사용)
- max_length: 각 전처리 방식에서 90% coverage를 만족하는 값
- 희소 빈도 처리를 하는 경우 min_freq = 3

평가 지표
- Accuracy
- Train Loss
- Validation Loss
- 학습 시간

비교 대상
1. 기본 전처리
   - 특수문자(.,!?^~;)를 공백으로 분리
2. 감성 표현 전처리
   - 연속 특수문자(.., ^^, ;;, ,, 등) 처리
   - 반복 자모(ㅋ, ㅎ, ㅠ, ㅜ 등) 처리
3. Character-level Tokenization
4. 형태소 분석
5. Subword Tokenization

---

### 2. 모델 비교

고정 조건
- 전처리: 전처리 비교에서 가장 성능이 좋았던 방법 사용
- 하이퍼파라미터: 초기값 사용 (필요시 튜닝된 값 사용)
- Dropout 미적용

Baseline
- Majority Baseline (Accuracy)

평가 지표
- Accuracy
- Train Loss
- Validation Loss
- 학습 시간

비교 대상
1. RNN
2. LSTM
3. GRU
4. Bi-LSTM
5. BERT

---

### 3. Dropout 효과

고정 조건
- 전처리: 전처리 비교에서 가장 성능이 좋았던 방법 사용
- 모델: 모델 비교에서 가장 성능이 좋았던 모델 사용
- 하이퍼파라미터: 초기값 사용 (필요시 튜닝된 값 사용)

평가 지표
- Accuracy
- Train Loss
- Validation Loss
- 학습 시간

비교 대상
1. Dropout 미적용
2. Dropout = 0.2
