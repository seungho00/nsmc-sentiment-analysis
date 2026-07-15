# Experiment Log

## Hyperparameter Tuning

- Model: LSTM
- Data: 기본 전처리 (특수문자 `.,!?^~;` 공백 분리)
- Max Length: 5

| embedding_dim | hidden_size | Best Epoch | Best Valid Loss | Test Accuracy |
|-:|-:|-:|-:|-:|
| 128 | 64 | 2 | 0.4841 | 0.7362 |
| 64 | 64 | 3 | 0.4930 | 0.7365 |
| 128 | 32 | 2 | 0.4878 | 0.7357 |
| 64 | 32 | 2 | 0.4888 | 0.7368 |
| **32** | **32** | **4** | **0.4921** | **0.7444** |
| 64 | 16 | 2 | 0.4914 | 0.7402 |
| **32** | **16** | **4** | **0.4944** | **0.7481** |

최종 하이퍼파라미터
```
embedding_dim = 32
hidden_size = 32
```
- embedding_dim=32, hidden_size=16에서 가장 높은 테스트 정확도(0.7481)을 기록하였다.
- 다만 모델 간(RNN, LSTM, GRU, Bi-LSTM) 비교 실험에서는 과도한 하이퍼파라미터 최적화를 피하고 공정한 비교를 위해 hidden_size를 32로 결정하였다.
- 또한 hidden_size를 64에서 32로 줄여도 성능 저하가 거의 없었으며, 모델 크기와 연산량을 줄일 수 있다는 장점도 있었다.


## MAX_LENGTH Candidate Test

- Model: LSTM
- Embedding Dimension: 32
- Hidden size: 32
- Data: 기본 전처리 (특수문자 `.,!?^~;` 공백 분리)

| Length | Coverage
|-:|-:|
| 5 | 0.333291 |
| 7 | 0.491830 |
| 12 | 0.757219 |
| 16 | 0.853002 |
| 21 | 0.904382 |

| max_length | Best Epoch | Best Valid Loss | Test Accuracy |
|-:|-:|-:|-:|
| 5 | 4 | 0.4921 | 0.7444 |
| 7 | 3 | 0.4576 | 0.7642 |
| 12 | 4 | 0.4184 | 0.7905 |
| 16 | 4 | 0.4047 | 0.8026 |
| 21 | 5 | 0.4080 | 0.8046 |

이후 실험에 사용할 coverage: 90%

- 토큰화 방식에 따라 문장 길이 분포가 달라지므로 max_length 대신 coverage를 고정 조건으로 사용하였다.
- 기본 전처리에서 약 90% coverage에 해당하는 max_length=21이 가장 높은 테스트 정확도(0.8046)를 기록하였다.
- 이후 전처리 비교 실험에서는 각 전처리 방식에서 90% coverage를 만족하는 max_length를 사용한다.



## 실험 1. 전처리 비교

고정 조건
- Model: LSTM
- Embedding dimension: 32
- Hidden size: 32
- max_length: 각 전처리 방식에서 90% coverage를 만족하는 값
- Min frequency = 3

### 기본 전처리
전처리 방식:
- 기본 띄어쓰기와 특수문자 `.,!?^~;` 공백 분리하여 토큰화

### 감성 표현 전처리
전처리 방식:
- 기본 전처리
- 연속 특수문자 `.., ^^, ;;, ,,` 처리
- 반복 자모 `ㅋㅋ, ㅎㅎ, ㄷㄷ, ㅠㅜ, ㅜㅜ, ㅡㅡ` 처리

### Character-level 전처리
전처리 방식:
- 문자 단위로 토큰화

### 형태소 분석 전처리
전처리 방식:
- 형태소 단위로 전처리
- Mecab 분석기 사용

### 결과
| Tokenizer | max_length | Best Epoch | Best Valid Loss | Test Accuracy |
|:-|-:|-:|-:|-:|
| Basic | 21 | 5 | 0.4080 | 0.8046 |
| Sentiment | 19 | 3 | 0.4064 | 0.8016 |
| character-level | 75 | 9 | 0.3478 | 0.8447 |
| morpheme | 39 | 4 | 0.3266 | 0.8546 |