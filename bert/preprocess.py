import pandas as pd
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import (
    VALID_SIZE,
    RANDOM_STATE,
    TOKENIZER,
    RAW_DIR,
)



## 데이터를 불러오는 함수 ##

train_file_path = RAW_DIR / "ratings_train.txt"
test_file_path = RAW_DIR / "ratings_test.txt"

def load_raw_data():
    try:
        train_df = pd.read_csv(train_file_path, sep='\t') # 탭으로 구분된 파일임을 명시
        print('train 데이터 로드 성공')
    except FileNotFoundError:
        print(f"오류: {train_file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
        exit()
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다

    try:
        test_df = pd.read_csv(test_file_path, sep='\t') # 탭으로 구분된 파일임을 명시
        print('test 데이터 로드 성공')
    except FileNotFoundError:
        print(f"오류: {test_file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
        exit()
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다

    return train_df, test_df



## 데이터 정제 함수 ##

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



## validation set 분리 함수 ##

def split_train_valid(train, column="label"):
    train, valid = train_test_split(
        train,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=train[column]
    )

    return train, valid



## 토큰화 함수 ##

def tokenize_data(df):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    tokenized_df = tokenizer(df.tolist())

    return tokenized_df



## 전처리 실행 ##

if __name__ == '__main__':
    train_df, test_df = load_raw_data()
    train_df = clean_data(train_df, "train")
    test_df = clean_data(test_df, "test")

    train_df, valid_df = split_train_valid(train_df)

    train_tokenized = tokenize_data(train_df["document"])


    lengths = [len(ids) for ids in train_tokenized["input_ids"]]

    length_counts = pd.Series(Counter(lengths)).sort_index()

    coverage = length_counts.cumsum() / len(lengths)

    print("mean length:", sum(lengths) / len(lengths))
    print("mode length:", length_counts.idxmax())
    print("50% coverage:", coverage[coverage >= 0.5].index[0])
    print("90% coverage:", coverage[coverage >= 0.9].index[0])
    print("95% coverage:", coverage[coverage >= 0.95].index[0])
    print("max length:", length_counts.index[-1])