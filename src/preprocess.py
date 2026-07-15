import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle

from config import TOKENIZER
from tokenizer_modules import (
    basic,
    sentiment,
    character_level,
    morpheme
)

BASE_DIR = Path(__file__).resolve().parent
train_file_path = BASE_DIR / '../data/raw/ratings_train.txt'
test_file_path = BASE_DIR / '../data/raw/ratings_test.txt'


# 데이터 불러오기
try:
    df_train = pd.read_csv(train_file_path, sep='\t') # 탭으로 구분된 파일임을 명시
    print('train 데이터 로드 성공')
except FileNotFoundError:
    print(f"오류: {train_file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다

try:
    df_test = pd.read_csv(test_file_path, sep='\t') # 탭으로 구분된 파일임을 명시
    print('test 데이터 로드 성공')
except FileNotFoundError:
    print(f"오류: {test_file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다



## 데이터 정제 ##

def remove_rows(df, mask, name, target):
    if mask.any():
        data_cnt = mask.sum()
        df = df[~mask]
        print(f"\n{name} 데이터의 {target} {data_cnt}개 제거 완료")

    return df

def clean_data(df, name):
    df = remove_rows(df, df["document"].isna(), name, "결측값")
    df = remove_rows(df, df["document"] == "", name, "빈 문자열")
    df = remove_rows(df, df["document"].str.strip() == "", name, "공백만 있는 문자열")

    return df

df_train = clean_data(df_train, "train")
df_test = clean_data(df_test, "test")



## validation set 분리 ##

df_train, df_valid = train_test_split(
    df_train,
    test_size=0.1,
    random_state=42,
    stratify=df_train["label"]
)



## 문자열 토큰화

TOKENIZERS = {
    "basic": basic.basic_tokenizer,
    "sentiment": sentiment.sentiment_tokenizer,
    "character_level": character_level.character_level_tokenizer,
    "morpheme": morpheme.morpheme_tokenizer
}
tokenizer = TOKENIZERS[TOKENIZER]

tokenized_documents_train = tokenizer(df_train, "document")
tokenized_documents_valid = tokenizer(df_valid, "document")
tokenized_documents_test = tokenizer(df_test, "document")



## word_to_id 생성 ##

# 희소 단어 처리를 위한 단어 개수 세기
word_count = Counter()

for words in tokenized_documents_train:
    word_count.update(words)


# 사전 정의, <UNK> 와 <PAD> 미리 입력
min_freq = 3
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
for word, count in word_count.items():
    if count >= min_freq:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word



## corpus 생성 ##

# train 데이터 corpus 생성
corpus_train = [
    [word_to_id.get(word, unk_id) for word in words] for words in tokenized_documents_train
    ]

# valid 데이터 corpus 생성
corpus_valid = [
    [word_to_id.get(word, unk_id) for word in words] for words in tokenized_documents_valid
    ]

# test 데이터 corpus 생성
corpus_test = [
    [word_to_id.get(word, unk_id) for word in words] for words in tokenized_documents_test
    ]



## padding ##

# 토큰 시퀀스 최대 길이를 구하기 위한 최빈값 탐색
length_counts = tokenized_documents_train.apply(len).value_counts()

# max_length 구하기
length_counts = length_counts.sort_index()
coverage = length_counts.cumsum() / length_counts.sum()
sequence_maxlen = coverage[coverage >= 0.9].index[0]
print("\nmax_length:",sequence_maxlen)


# train padding 실행
corpus_pad_train = np.zeros((len(corpus_train), sequence_maxlen), dtype=int)

for i, corpus in enumerate(corpus_train):
    length = min(len(corpus), sequence_maxlen)
    corpus_pad_train[i, :length] = corpus[:length]

# valid padding 실행
corpus_pad_valid = np.zeros((len(corpus_valid), sequence_maxlen), dtype=int)

for i, corpus in enumerate(corpus_valid):
    length = min(len(corpus), sequence_maxlen)
    corpus_pad_valid[i, :length] = corpus[:length]

# test padding 실행
corpus_pad_test = np.zeros((len(corpus_test), sequence_maxlen), dtype=int)

for i, corpus in enumerate(corpus_test):
    length = min(len(corpus), sequence_maxlen)
    corpus_pad_test[i, :length] = corpus[:length]



## pickle 저장 ##

# 저장 위치
SAVE_BASE_DIR = BASE_DIR / "../data/preprocessed"

# label 저장
datasets = [(df_train, "train"), (df_valid, "valid"), (df_test, "test")]
for data, name in datasets:
    save_path = SAVE_BASE_DIR / f"label_{name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(data["label"].to_numpy(), f)


# word_to_id, id_to_word 저장
save_name = "word_to_id.pkl"
save_path = SAVE_BASE_DIR / save_name
with open(save_path, "wb") as f:
    pickle.dump(word_to_id, f)

save_name = "id_to_word.pkl"
save_path = SAVE_BASE_DIR / save_name
with open(save_path, "wb") as f:
    pickle.dump(id_to_word, f)


# corpus_pad 저장
datasets = [(corpus_pad_train, "train"),
            (corpus_pad_valid, "valid"),
            (corpus_pad_test, "test")]
for data, name in datasets:
    save_path = SAVE_BASE_DIR / f"corpus_pad_{name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(data, f)