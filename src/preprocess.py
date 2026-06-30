import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
train_file_path = BASE_DIR / '../data/ratings_train.txt'
test_file_path = BASE_DIR / '../data/ratings_test.txt'


# 데이터 불러오기
try:
    df_train = pd.read_csv(train_file_path, sep='\t') # 탭으로 구분된 파일임을 명시
    print('train 데이터 로드 성공')
    # print(df_train.head())
except FileNotFoundError:
    print(f"오류: {train_file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다

try:
    df_test = pd.read_csv(test_file_path, sep='\t') # 탭으로 구분된 파일임을 명시
    print('test 데이터 로드 성공')
    # print(df_test.head())
except FileNotFoundError:
    print(f"오류: {test_file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다



## 결측값 제거 ##

# 결측값 개수
print("\ntrain 데이터의 결측값의 수:", df_train["document"].isna().sum())
print("test 데이터의 결측값의 수:", df_test["document"].isna().sum())

# # 결측값 출력
# print("\ntrain 데이터의 결측값 목록")
# print(df_train[df_train["document"].isna()])

# print("\ntest 데이터의 결측값 목록")
# print(df_test[df_test["document"].isna()])

# 빈 문자열 확인
print("\ntrain 데이터의 빈 문자열 목록")
print(df_train[df_train["document"] == ""])

print("\ntest 데이터의 빈 문자열 목록")
print(df_test[df_test["document"] == ""])

# 공백만 있는 문자열 확인
print("\ntrain 데이터의 공백만 있는 문자열 목록")
print(df_train[df_train["document"].str.strip() == ""])

print("\ntest 데이터의 공백만 있는 문자열 목록")
print(df_test[df_test["document"].str.strip() == ""])

# 결측값 제거
print("\ntrain 데이터의 결측값 삭제")
print("삭제 전:", len(df_train))

df_train = df_train.dropna(subset=["document"])

print("삭제 후:", len(df_train))
print("결측값의 수:", df_train["document"].isna().sum())

print("\ntest 데이터의 결측값 삭제")
print("삭제 전:", len(df_test))

df_test = df_test.dropna(subset=["document"])

print("삭제 후:", len(df_test))
print("결측값의 수:", df_test["document"].isna().sum(), "\n")



## 문자열 전처리 ##

# 특수 문자도 따로 처리 될 수 있도록 공백 추가
df_train["document"] = df_train["document"].str.replace(
    r"([.,!?])", r" \1 ", regex=True
)
df_test["document"] = df_test["document"].str.replace(
    r"([.,!?])", r" \1 ", regex=True
)

# 문자열 토큰화
tokenized_documents_train = df_train["document"].str.split()
tokenized_documents_test = df_test["document"].str.split()



## word_to_id 생성 ##

# 사전 정의, <UNK> 와 <PAD> 미리 입력
pad_id = 0
unk_id = 1
word_to_id = {
    "<PAD>" : pad_id,
    "<UNK>" : unk_id
}
id_to_word = {
    pad_id : "<PAD>",
    unk_id : "<UNK>"
}

# 구현
for words in tokenized_documents_train:
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

for i in range(min(15, len(id_to_word))):
    print(i,':', id_to_word[i])
print("")


## corpus 생성 ##

# train 데이터 corpus 생성
train_corpus = [
    [word_to_id[word] for word in words] for words in tokenized_documents_train
    ]
# print(train_corpus[:5])

# test 데이터 corpus 생성
test_corpus = [
    [word_to_id.get(word, unk_id) for word in words] for words in tokenized_documents_test
    ]
# print(test_corpus[:5])



## padding ##

# 토큰 시퀀스 최대 길이를 구하기 위한 최빈값 탐색
length_counts = tokenized_documents_train.apply(len).value_counts()

print(length_counts.head(8), end="\n\n")  # 가장 많은 길이들
# document
# 5     12598
# 6     12565
# ...
# 9      8907
# 10     7986
# Name: count, dtype: int64

# mode_value = length_counts.idxmax()  # 가장 큰 value의 index를 반환
# print("mode:", mode_value)

# 각 길이별 커버리지 구하기 (각 길이에서 잘리지 않는 문서의 비율)
length_counts = length_counts.sort_index()
coverage = length_counts.cumsum() / length_counts.sum()
print(coverage[(coverage > 0.33) & (coverage <= 0.9)])
# document
# 5     0.333291
# ...
# 7     0.491830
# ...
# 12    0.757219
# ...
# 16    0.853002
# ...
# 20    0.898610
# Name: count, dtype: float64

# 토큰 시퀀스 최대 길이 후보군
candidates = [[5, 7, 12, 16, 20]]

# padding 실행



## validation set 분리 ##