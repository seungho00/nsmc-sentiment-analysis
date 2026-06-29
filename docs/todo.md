# TODO (NSMC Sentiment Analysis)

---

## 0. 데이터

- [x] NSMC 데이터셋 다운로드
- [x] train / test 분리

---

## 1. 전처리

- [x] word_to_id 생성
- [ ] corpus 생성
- [ ] unknown token 처리 (OOV handling)
- [ ] padding 구현 (fixed sequence length)
- [ ] validation set 분리 (train에서 split)

---

## 2. Baseline (기준 성능)

- [ ] Majority baseline 계산 및 평가 (accuracy)

---

## 3. 모델 구현

- [ ] Embedding layer 구현
- [ ] LSTM 모델 구현
- [ ] 학습 루프 작성 (train / eval 분리)
- [ ] 모델 저장 및 로드 기능 구현

---

## 4. 실험

### 데이터 전처리 관련
- [ ] word_to_id 모든 특수 문자를 처리하기 (@, ^, ~이나 .., ㅋㅋ, ㅠㅠ 등등)
- [ ] word_to_id 최소 빈도 단어 처리
- [ ] word_to_id character 단위
- [ ] word_to_id 형태소 분석
- [ ] word_to_id subword 방식

- [ ] LSTM baseline 모델 학습
- [ ] Dropout 추가 실험
- [ ] GRU 모델 구현 및 실험
- [ ] Bi-LSTM 모델 구현 및 실험
- [ ] BERT 모델과 성능 비교

---

## 5. 평가 및 분석

- [ ] Confusion Matrix 출력 (모든 모델)
- [ ] Accuracy / F1-score 비교 정리
- [ ] Loss / Accuracy 그래프 저장
- [ ] majority baseline vs 모델 성능 비교

---

## 6. 문서화

- [ ] README 작성 (데이터, 모델 구조, 사용 방법)
- [ ] 실험 결과 정리 (table + 분석)