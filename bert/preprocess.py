import pandas as pd
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import (
    VALID_SIZE,
    RANDOM_STATE,
    TOKENIZER,
    RAW_DATA_DIR,
    PREPROCESSED_DIR,
)



## 데이터를 불러오는 함수 ##

train_file_path = RAW_DATA_DIR / "ratings_train.txt"
test_file_path = RAW_DATA_DIR / "ratings_test.txt"

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

def split_train_valid(train, valid_size=VALID_SIZE, random_state=RANDOM_STATE, stratify_column="label"):
    train, valid = train_test_split(
        train,
        test_size=valid_size,
        random_state=random_state,
        stratify=train[stratify_column]
    )

    return train, valid



## max_length 탐색 함수 ##

def get_maxlength(input_ids, ratio):
    lengths = [len(ids) for ids in input_ids]
    
    length_counts = pd.Series(Counter(lengths)).sort_index()

    coverage = length_counts.cumsum() / len(lengths)

    return coverage[coverage >= ratio].index[0]



## 패딩 후 데이터셋 반환 ##

def get_padded_dataset(tokenized_data, max_length, pad_token_id):
    n = len(tokenized_data["input_ids"])

    input_ids = torch.full(
        (n, max_length),
        fill_value=pad_token_id,
        dtype=torch.long
    )
    attention_mask = torch.zeros((n, max_length), dtype=torch.long)
    token_type_ids = torch.zeros((n, max_length), dtype=torch.long)


    for idx, ids in enumerate(tokenized_data["input_ids"]):
        ids = torch.tensor(ids, dtype=torch.long)
        length = min(max_length, len(ids))

        input_ids[idx, :length-1] = ids[:length-1]
        input_ids[idx, length-1] = ids[-1]                   # [SEP] 토큰 유지

        attention_mask[idx, :length] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }



## 전처리 실행 ##

if __name__ == '__main__':

    # 데이터 로드 및 정제
    train_df, test_df = load_raw_data()
    train_df = clean_data(train_df, "train")
    test_df = clean_data(test_df, "test")


    # 검증 데이터 분리
    train_df, valid_df = split_train_valid(train_df)


    # 토큰화
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    train_tokenized = tokenizer(train_df["document"].tolist())
    valid_tokenized = tokenizer(valid_df["document"].tolist())
    test_tokenized = tokenizer(test_df["document"].tolist())


    # max_length 찾기
    max_length = get_maxlength(train_tokenized["input_ids"], 0.95)


    # 데이터셋 만들기
    train_encodings = get_padded_dataset(
        train_tokenized,
        max_length,
        tokenizer.pad_token_id
    )

    valid_encodings = get_padded_dataset(
        valid_tokenized,
        max_length,
        tokenizer.pad_token_id
    )

    test_encodings = get_padded_dataset(
        test_tokenized,
        max_length,
        tokenizer.pad_token_id
    )

    # 저장
    data_group = [
        ('train', train_encodings, train_df['label']),
        ('valid', valid_encodings, valid_df['label']),
        ('test', test_encodings, test_df['label'])
    ]
    for name, corpus, label in data_group:
        corpus_save_path = PREPROCESSED_DIR / f'{name}_encodings.pt'
        torch.save(corpus, corpus_save_path)

        label_save_path = PREPROCESSED_DIR / f'{name}_label.pt'
        torch.save(torch.tensor(label.to_numpy(), dtype=torch.long), label_save_path)
    print("\n저장 완료")