# Experiment Log

## Majority Baseline

- 목적: 모델 학습 전 최소 성능 기준 확인
- 방식: 가장 많은 클래스(긍정/부정)를 항상 예측
- Validation Accuracy: 50.11%
- Test Accuracy: 49.65%

### Interpretation
- 데이터 클래스 비율이 거의 균형이므로 majority baseline은 약 50%의 정확도를 보임.
- 이후 모델은 최소한 이 성능 이상을 달성해야 의미 있는 학습 결과로 판단할 수 있음.


## Prototype Model

- Data: 기본 전처리 (특수문자 `.,!?^~;` 공백 분리)
- Max Length: 5

- Model: LSTM
- Embedding Dimension: 128
- Hidden size: 64

- **Test Accuracy: 73.62%**

### Comparison
- Majority baseline 대비 +23.97%p 향상
- 문장 정보를 활용해 긍정/부정 분류가 가능함을 확인



## Hyperparameter Tuning

- Model: LSTM
- Data: 기본 전처리 (특수문자 `.,!?^~;` 공백 분리)
- Max Length: 5
- Best validation loss 기준 모델 저장(Model Checkpointing)

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
- Best validation loss 기준 모델 저장 (Model Checkpointing)

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
- Best validation loss 기준 모델 저장 (Model Checkpointing)

평가 지표
- Best Validation Loss
- Test Accuracy

### 1. 기본 전처리
전처리 방식:
- 기본 띄어쓰기와 특수문자 `.,!?^~;` 공백 분리하여 토큰화
- Min frequency: 3

### 2. 감성 표현 전처리
전처리 방식:
- 기본 전처리
- 연속 특수문자 `.., ^^, ;;, ,,` 처리
- 반복 자모 `ㅋㅋ, ㅎㅎ, ㄷㄷ, ㅠㅜ, ㅜㅜ, ㅡㅡ` 처리
- Min frequency: 3

### 3. Character-level 전처리
전처리 방식:
- 문자 단위로 토큰화

### 4. 형태소 분석 전처리
전처리 방식:
- 형태소 단위로 전처리
- Mecab 분석기 사용
- Min frequency: 3

### 5. subword 분석 전처리
전처리 방식:
- SentencePiece 기반 subword tokenization
- Model type: Unigram
- Vocabulary size: 8,192
- Vocabulary 생성 시 hard_vocab_limit=False 적용


### 결과
| Tokenizer | vocab_size | max_length | Best Epoch | Best Valid Loss | Test Accuracy |
|:-|-:|-:|-:|-:|-:|
| Basic | 39,510 | 21 | 4 | 0.4026 | 0.8078 |
| Sentiment | 39,520 | 19 | 5 | 0.4084 | 0.8092 |
| Character-level | 2,939 | 75 | 9 | 0.3478 | 0.8471 |
| Morpheme | 21,449 | 39 | 7 | 0.3276 | 0.8578 |
| Subword | 8,192 | 37 | 6 | 0.3395 | 0.8488 |


- 형태소 분석(Mecab)이 가장 높은 정확도(85.78%)를 기록하였다. 한국어의 조사와 어미를 분리하여 의미 단위를 보존한 것이 성능 향상에 기여했을 가능성이 있다.
- Character-level과 Subword는 형태소 분석과 비슷한 수준의 성능을 보였다. 특히 Character-level은 가장 작은 vocabulary(2,939)만으로도 높은 정확도(84.71%)를 달성하여 OOV 문제에 강한 특성을 확인할 수 있었다.
- Basic 전처리는 다른 전처리 방식보다 낮은 성능을 보였다. 단순 띄어쓰기 기반 토큰화만으로는 한국어의 형태 변화와 의미를 충분히 반영하지 못한 것으로 해석할 수 있다.
- 감성 표현 전처리는 Basic보다 정확도가 0.14%p 높았지만 차이가 매우 작았다. 따라서 추가한 규칙이 성능 향상에 기여했을 가능성은 있으나, 본 실험만으로 유의미한 개선이라고 결론 내리기는 어렵다.
- Subword는 Character-level보다 약 절반 수준의 시퀀스 길이(max_length 37 vs. 75)에서 약간 더 높은 정확도(84.88% vs. 84.71%)를 기록하여, 시퀀스 길이 측면에서 효율적인 토큰화 방식임을 확인할 수 있었다.
- 전반적으로 한국어 감성 분석에서는 단순 띄어쓰기 기반 토큰화인 Basic보다 Morpheme, Character-level, Subword 토큰화 방식이 더 효과적인 것으로 나타났다. 이는 한국어의 형태적 특성과 의미 정보를 보다 효과적으로 반영하는 토큰화 방식이 감성 분석 성능 향상에 도움이 될 가능성을 시사한다.


## 실험 2. 모델 비교

고정 조건
- 전처리: 형태소 분석 (Mecab)
- Min frequency: 3
- max_length: 39 (형태소 분석에서 90% coverage)
- Embedding dimension: 32
- Hidden size: 32 (양방향 모델에서는 단방향 모델과 동일한 출력 차원을 맞추기 위해 절반으로 설정)
- Dropout 미적용
- Epoch: 10
- Best validation loss 기준 모델 저장 (Model Checkpointing)

평가 지표
- Best Validation Loss
- Test Accuracy
- F1-score
- Confusion Matrix
- Train/Validation Loss 그래프

비교 모델
- RNN
- LSTM
- GRU
- BiLSTM
- BiGRU

### 결과
(각 모델의 loss, acc 그래프와 confusion matrix의 이미지는 `results/experiment2`폴더의 각 모델 폴더 안에서 볼 수 있다)

| Model | Best Epoch | Best Valid Loss | Test Accuracy | F1-score |
|:-|-:|-:|-:|-:|
| RNN | 10 | 0.5636 | 0.7420 | 0.7299 |
| LSTM | 5 | 0.3226 | 0.8574 | 0.8580 |
| GRU | 4 | 0.3210 | 0.8577 | 0.8573 |
| BiLSTM | 3 | 0.3314 | 0.8535 | 0.8580 |
| BiGRU | 4 | 0.3257 | 0.8568 | 0.8567 |

Confusion Matrix (True/Predicted)
| Model | N/N | N/P | P/N | P/P | 
|:-|-:|-:|-:|-:|
| RNN | 19656 | 5170 | 7734 | 17437 |
| LSTM | 21331 | 3495 | 3634 | 21537 |
| GRU | 21503 | 3323 | 3792 | 21379 |
| BiLSTM | 20535 | 4291 | 3035 | 22136 |
| BiGRU | 21449 | 3377 | 3781 | 21390 |

- RNN은 성능과 학습 효율 모두 가장 낮은 결과를 보였다. 다른 모델들은 3~5 epoch에서 최적 성능에 도달한 반면, RNN은 10 epoch까지도 충분히 수렴하지 않았으며 Test Accuracy 역시 크게 낮았다.
- 단방향 모델과 양방향 모델 간의 성능 차이는 매우 작았다. BiLSTM과 BiGRU가 일부 지표에서는 소폭 낮은 성능을 보였지만, 전체적으로는 오차 범위 내의 차이로 판단되어 양방향 구조가 성능 향상에 유의미한 영향을 주었다고 보기 어려웠다.
- Confusion Matrix의 결과도 RNN을 제외하면 큰 차이를 보이지 않았다. 다만 BiLSTM은 Negative를 Positive로 오분류하는 비율(N/P)이 다소 증가한 반면, Positive를 Negative로 오분류하는 비율(P/N)은 가장 낮았다. 즉 Positive 클래스에 다소 치우친 예측 경향을 보였으나, 전체 성능에는 큰 차이를 만들지 않았다.
- 전체적인 성능이 거의 동일하다면 구조가 비교적 단순하고 계산량이 적은 GRU를 사용하는 것이 효율적이라고 판단하였다. 따라서 실험 3(Dropout 효과)에서는 GRU를 기준 모델로 사용하였다.


## 실험 3. Dropout 효과

조건
- dropout을 제외한 실험 2의 고정 조건과 동일
- 실험 2의 GRU 모델 사용

평가 지표
- Best Validation Loss
- Test Accuracy
- F1-score
- Confusion Matrix
- Train/Validation Loss 그래프

비교 dropout_rate
- 0.0
- 0.2

### 결과

| Dropout rate | Best Epoch | Best Valid Loss | Test Accuracy | F1-score |
|-:|-:|-:|-:|-:|-:|-:|
| 0.0 | 4 | 0.3210 | 0.8577 | 0.8573 |
| 0.2 | 5 | 0.3296 | 0.8576 | 0.8588 |

Confusion Matrix (True/Predicted)
| Model | N/N | N/P | P/N | P/P | 
|:-|-:|-:|-:|-:|
| 0.0 | 21503 | 3323 | 3792 | 21379 |
| 0.2 | 21221 | 3605 | 3517 | 21654 |

- Dropout 0.2를 적용했을 때 Test Accuracy는 거의 동일했으며, F1-score는 소폭 증가하였다. 하지만 Best Validation Loss는 Dropout을 적용하지 않은 모델이 더 낮은 값을 기록하였다. 다만 두 결과의 차이는 크지 않아, 본 실험 환경에서는 Dropout의 뚜렷한 성능 향상 효과를 확인하지 못했다.

- Confusion Matrix를 비교하면 Dropout 0.2를 적용한 모델은 Dropout을 적용하지 않은 모델보다 Positive 예측 비율이 증가하였다. 이로 인해 실제 Positive 중 올바르게 예측한 비율인 Positive Recall은 증가했지만, Negative를 Positive로 잘못 판단하는 경우(False Positive)도 증가하였다. Precision은 감소했지만 Recall 증가 폭이 더 커지면서 F1-score는 소폭 향상되었다. 하지만 전체적인 분류 성능 향상으로 이어지지는 않았다.

- Dropout이 큰 효과를 발휘하지 못한 이유 중 하나는, 모델 학습 과정에서 Validation Loss를 기준으로 최적 모델을 저장하는 checkpointing을 적용하고 있기 때문으로 판단된다. Checkpointing을 통해 과적합이 진행되기 이전의 모델을 선택하기 때문에, 과적합이 심화된 후반 모델의 성능 차이가 최종 평가에 반영되지 않았다. 따라서 Dropout의 차이가 최종 평가에서 작게 보였을 가능성이 있다.

- Train/Validation Loss 그래프를 비교했을 때 Dropout 0.0과 0.2 모델은 유사한 학습 패턴을 보였다. Dropout 0.2를 적용한 모델에서 과적합 발생 시점이 1 epoch 늦춰지는 경향은 있었지만, Validation Loss의 감소 폭과 이후 증가 양상이 크게 다르지 않았다. 따라서 checkpointing을 사용하지 않고 마지막 epoch의 모델을 사용하더라도 Dropout으로 인한 성능 차이는 크지 않았을 것으로 예상된다.

- Dropout이 큰 효과를 발휘하지 못한 또 다른 이유로는, 본 실험의 GRU 모델은 약 70만 개의 학습 가능한 파라미터를 가지는 비교적 단순한 구조로 구성되어 있어 대규모 모델에 비하면 Dropout을 통한 규제 효과가 제한적이었을 가능성이 있다. 일반적으로 모델의 파라미터 수가 크고 과적합 위험이 높은 경우 Dropout의 효과가 크게 나타날 수 있지만, 본 실험의 GRU 모델에서는 추가적인 규제가 성능 향상으로 이어지지 않은 것으로 판단된다.