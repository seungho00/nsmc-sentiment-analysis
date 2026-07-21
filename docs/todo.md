# TODO (NSMC Sentiment Analysis)

---

## 0. 모델 설계

- [x] 문제 정의
- [x] 모델 구조 설계
- [x] 계층별 tensor shape 설계
- [x] 하이퍼파라미터 설계
- [x] 실험 설계

---

## 1. 전처리

- [x] NSMC 데이터셋 다운로드
- [x] train / test 분리
- [x] word_to_id 생성
- [x] 희소 빈도 처리
- [x] corpus 생성
- [x] unknown token 처리 (OOV handling)
- [x] padding 구현 (fixed sequence length)
- [x] validation set 분리 (train에서 split)
- [x] 전처리 결과 pickle 저장 및 불러오기 구현

---

## 2. Baseline (기준 성능)

- [x] Majority baseline 계산 및 평가 (accuracy)

---

## 3-1. 모델 구현

- [x] Embedding layer 구현
- [x] LSTM 모델 구현
- [x] 학습 루프 작성 (train / eval 분리)
- [x] 모델 저장 및 로드 기능 구현
- [x] majority baseline vs 모델 성능 비교

---

## 3-2. 실험 환경 구축

- [x] 하이퍼파라미터 튜닝
- [x] max_length 후보군 테스트
- [x] 전처리 모듈화

---

## 4. 실험

### 실험 1
- [x] 감성 표현 전처리
- [x] Character-level Tokenization
- [x] 형태소 분석
- [x] subword Tokenization

### 실험 2 (구현)
- [x] RNN 모델 구현 및 실험
- [x] GRU 모델 구현 및 실험
- [x] Bi-LSTM 모델 구현 및 실험
- [x] Bi-GRU 모델 구현 및 실험

### 실험 2 (평가)
- [x] Confusion Matrix 출력 (모든 모델)
- [ ] Accuracy / F1-score 비교 정리
- [x] Loss / Accuracy 그래프 저장

### 실험 3
- [ ] Dropout 구현 및 실험

### 실험 4
- [ ] BERT 모델 구현 및 다른 모델들과 성능 비교

---

## 5. 문서화

- [ ] README 작성 (데이터, 모델 구조, 사용 방법)
- [ ] 실험 결과 정리 (table + 분석)