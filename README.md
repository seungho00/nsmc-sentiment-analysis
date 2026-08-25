# nsmc-sentiment-analysis
Sequence 모델(RNN, LSTM, GRU, BiLSTM, BiGRU)과 다양한 한국어 전처리 기법을 비교한 NSMC 감성 분석 프로젝트

## 1. 프로젝트 소개

- 목적
  - 책으로 학습한 딥러닝 이론을 직접 구현하며 PyTorch 사용법과 프로젝트 개발 경험을 쌓기 위함
  - 한국어 텍스트 감성 분석 모델 구현
  - 공부하면서 생긴 궁금증을 해결하기 위한 실험 진행

- Dataset
  - Naver Sentiment Movie Corpus (NSMC)
  - 프로젝트 실행을 위해 raw data는 직접 다운로드 후, data/raw에 위치시켜야 한다.

- 참고 자료
  - 『밑바닥부터 시작하는 딥러닝』
  - 『파이썬 딥러닝 머신러닝 입문』
  - ChatGPT (개념 학습, 코드 리뷰 및 개발 보조)

## 2. 개발 환경

- **OS**: macOS
- **학습 환경**
  - RNN, LSTM, GRU, BiLSTM, BiGRU: macOS
  - BERT: Google Colab (NVIDIA T4 GPU)
- **의존성 패키지**: `requirements.txt` 참고

## 3. 프로젝트 구조
```
nsmc-sentiment-analysis
├── bert                                        # BERT 모델 학습, 평가 코드 및 결과 데이터
├── data                                        # 데이터셋
├── docs                                        # 실험 설계, 일지, 보고서
├── results                                     # 실험 결과 데이터
├── src
│   ├── models                                  # 모델 클래스
│   ├── tokenizer_modules                       # 토크나이저 구현
│   ├── baseline.py                             # Majority baseline
│   ├── config.py
│   ├── data_utils.py                           # 전처리 후 데이터 로드
│   ├── evaluate.py                             # 모델 평가
│   ├── max_length_test.py                      # max_length 후보군 탐색 실험 코드
│   ├── preprocess.py                           # 데이터 전처리
│   ├── train_sentencepiece.py                  # SentencePiece 학습 코드
│   ├── train.py                                # 시퀀스 모델 학습 코드
│   └── visualization.py                        # 학습 과정 loss/acc 그래프 시각화 코드
├── README.md
└── requirements.txt
```

## 4. 모델 구조
(자세한 내용은 [`model_design.md`](docs/model_design.md)와 [`experiment_log.md`](docs/experiment_log.md) 참고)

Optimizer: Adam<br>

Input<br>
↓<br>
Embedding<br>
↓<br>
2-layer Sequence Model<br>
↓<br>
Linear (Affine, input: Last Hidden State)<br>
↓<br>
BCEWithLogitsLoss (Sigmoid, Binary Cross Entropy)

## 5. 실험 내용
(자세한 내용은 [`model_design.md`](docs/model_design.md)와 [`experiment_log.md`](docs/experiment_log.md) 참고)

### 0. 하이퍼파라미터 튜닝 및 MAX_LENGTH 탐색
- 책 『파이썬 딥러닝 머신러닝 입문』의 수치를 초기값으로 사용
- 이후 테스트를 진행하며 튜닝

---

### 1. 전처리 비교

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

비교 대상
1. RNN
2. LSTM
3. GRU
4. Bi-LSTM
5. Bi-GRU

---

### 3. Dropout 효과

비교 대상
1. Dropout 미적용
2. Dropout = 0.2

---

### 4. 추가 실험 (BERT)

비교 대상
- BERT
- 실험 2에서 가장 성능이 우수했던 모델

---

### 실험 결과 요약
(자세한 결과는 [`experiment_log.md`](docs/experiment_log.md) 참고)

#### 주요 결과
- 여러 전처리 방식 중 형태소 분석(Mecab)이 가장 높은 성능을 기록했다.
- GRU가 가장 낮은 Validation Loss와 가장 높은 Test Accuracy를 보였다.
- Dropout(0.2)은 성능 향상 효과가 거의 없었다.

#### 실험 1
- Model: LSTM
- Embedding dimension: 32
- Hidden size: 32
- max_length: 각 전처리 방식에서 90% coverage를 만족하는 값
- Best validation loss 기준 모델 저장 (Model Checkpointing)

| Tokenizer | vocab_size | max_length | Best Epoch | Best Valid Loss | Test Accuracy |
|:-|-:|-:|-:|-:|-:|
| Basic | 39,510 | 21 | 4 | 0.4026 | 0.8078 |
| Sentiment | 39,520 | 19 | 5 | 0.4084 | 0.8092 |
| Character-level | 2,939 | 75 | 9 | 0.3478 | 0.8471 |
| **Morpheme** | 21,449 | 39 | 7 | 0.3276 | **0.8578** |
| Subword | 8,192 | 37 | 6 | 0.3395 | 0.8488 |

#### 실험 2
- 전처리: 형태소 분석 (Mecab)
- Min frequency: 3
- max_length: 39 (형태소 분석에서 90% coverage)
- Embedding dimension: 32
- Hidden size: 32 (양방향 모델에서는 단방향 모델과 동일한 출력 차원을 맞추기 위해 절반으로 설정)
- Dropout 미적용
- Epoch: 10
- Best validation loss 기준 모델 저장 (Model Checkpointing)

| Model | Best Epoch | Best Valid Loss | Test Accuracy | F1-score |
|:-|-:|-:|-:|-:|
| RNN | 10 | 0.5636 | 0.7420 | 0.7299 |
| LSTM | 5 | 0.3226 | 0.8574 | 0.8580 |
| GRU | 4 | 0.3210 | 0.8577 | 0.8573 |
| BiLSTM | 3 | 0.3314 | 0.8535 | 0.8580 |
| BiGRU | 4 | 0.3257 | 0.8568 | 0.8567 |

<br>

Confusion Matrix (True/Predicted)
| Model | N/N | N/P | P/N | P/P | 
|:-|-:|-:|-:|-:|
| RNN | 19656 | 5170 | 7734 | 17437 |
| LSTM | 21331 | 3495 | 3634 | 21537 |
| GRU | 21503 | 3323 | 3792 | 21379 |
| BiLSTM | 20535 | 4291 | 3035 | 22136 |
| BiGRU | 21449 | 3377 | 3781 | 21390 |

#### 실험 3
- dropout을 제외한 실험 2의 고정 조건과 동일
- 실험 2의 GRU 모델 사용

| Dropout rate | Best Epoch | Best Valid Loss | Test Accuracy | F1-score |
|-:|-:|-:|-:|-:|
| 0.0 | 4 | 0.3210 | 0.8577 | 0.8573 |
| 0.2 | 5 | 0.3296 | 0.8576 | 0.8588 |

<br>

Confusion Matrix (True/Predicted)
| Model | N/N | N/P | P/N | P/P | 
|-:|-:|-:|-:|-:|
| 0.0 | 21503 | 3323 | 3792 | 21379 |
| 0.2 | 21221 | 3605 | 3517 | 21654 |

#### 실험 4
- Model: klue/bert-base
- Max length: 62
- Batch size: 32
- Epochs: 5
- Hidden Dropout: 0.1 (default)
- Attention Dropout: 0.1 (default)
- Learning rate: 2e-5
- Optimizer: AdamW
- Best validation loss 기준 모델 저장 (Model Checkpointing)

| Model | Best Epoch | Best Valid Loss | Test Accuracy | F1-score |
|:-|-:|-:|-:|-:|
| BERT | 2 | 0.2501 | 0.9026 | 0.9045 |
| GRU (dropout=0.2) | 5 | 0.3296 | 0.8576 | 0.8588 |

<br>

Confusion Matrix (True/Predicted)
| Model | N/N | N/P | P/N | P/P | 
|:-|-:|-:|-:|-:|
| BERT | 22078 | 2748 | 2120 | 23051 |
| GRU (dropout=0.2) | 21221 | 3605 | 3517 | 21654 |


## 사용 방법
1. `config.py`에서 실험 설정
2. `preprocess.py` 실행
3. `train.py` 실행
4. `visualization.py` 실행 (선택)
5. `evaluate.py` 실행
6. 실행 결과 `results/`에 자동 저장