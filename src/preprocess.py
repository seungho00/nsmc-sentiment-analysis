import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import sentencepiece as spm

from config import (
    VALID_SIZE,
    RANDOM_STATE,
    TOKENIZER,
    MIN_FREQ,
    PAD_ID,
    UNK_ID,
    PREPROCESSED_DIR,
    RAW_DIR,
    CHECKPOINTS_DIR,
)
from tokenizer_modules import (
    basic,
    sentiment,
    character_level,
    morpheme
)

train_file_path = RAW_DIR/ ' ratings_train.txt'
test_file_path = RAW_DIR / 'ratings_test.txt'


# 데이터 불러오는 함수
def load_raw_data():
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

    return df_train, df_test



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



## word_to_id 생성 함수##

def make_word_to_id(tokenized_documents):
    # 희소 단어 처리를 위한 단어 개수 세기
    word_count = Counter()

    for words in tokenized_documents:
        word_count.update(words)


    # 사전 정의, <UNK> 와 <PAD> 미리 입력
    word_to_id = {
        "<PAD>" : PAD_ID,
        "<UNK>" : UNK_ID
    }
    id_to_word = {
        PAD_ID : "<PAD>",
        UNK_ID : "<UNK>"
    }

    # 구현
    for word, count in word_count.items():
        if count >= MIN_FREQ:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    return word_to_id, id_to_word



## corpus 생성 함수 ##

def make_corpus(tokenized_documents, word_to_id):
    corpus = [
        [word_to_id.get(word, UNK_ID) for word in words] for words in tokenized_documents
    ]

    return corpus

def make_sp_corpus(sp, df, column="document"):
    return [
        sp.encode(sentence, out_type=int)
        for sentence in df[column]
    ]


## padding 함수 ##

def padding(corpora, max_length):
    corpus_pad = np.zeros((len(corpora), max_length), dtype=int)

    for i, corpus in enumerate(corpora):
        length = min(len(corpus), max_length)
        corpus_pad[i, :length] = corpus[:length]

    return corpus_pad



## 전처리 실행 ##

if __name__ == "__main__":
    
    # 데이터 불러오기
    df_train, df_test = load_raw_data()
    
    # 데이터 정제
    df_train = clean_data(df_train, "train")
    df_test = clean_data(df_test, "test")

    # 검증 데이터 분리
    df_train, df_valid = split_train_valid(df_train)



    ## 토큰화, vocab, corpus 생성 ##

    if TOKENIZER != "subword":
        
        # tokenizer 선택
        TOKENIZERS = {
            "basic": basic.basic_tokenizer,
            "sentiment": sentiment.sentiment_tokenizer,
            "character_level": character_level.character_level_tokenizer,
            "morpheme": morpheme.morpheme_tokenizer,
        }
        tokenizer = TOKENIZERS[TOKENIZER]

        # 문자열 토큰화
        tokenized_documents_train = tokenizer(df_train)
        tokenized_documents_valid = tokenizer(df_valid)
        tokenized_documents_test = tokenizer(df_test)


        # word_to_id 생성
        word_to_id, id_to_word = make_word_to_id(tokenized_documents_train)
        print("\nvocab_size:", len(word_to_id))

        # corpus 생성
        corpus_train = make_corpus(tokenized_documents_train, word_to_id)
        corpus_valid = make_corpus(tokenized_documents_valid, word_to_id)
        corpus_test = make_corpus(tokenized_documents_test, word_to_id)

    else:
        sp = spm.SentencePieceProcessor()

        sp.load(
            str(CHECKPOINTS_DIR / "tokenizer/sentencepiece.model")
        )

        corpus_train = make_sp_corpus(sp, df_train)
        corpus_valid = make_sp_corpus(sp, df_valid)
        corpus_test = make_sp_corpus(sp, df_test)

        print("\nvocab_size:", sp.get_piece_size())



    # 토큰 시퀀스 최대 길이를 구하기 위한 최빈값 탐색
    length_counts = pd.Series(
        [len(sequence) for sequence in corpus_train]
    ).value_counts()

    # max_length 구하기
    length_counts = length_counts.sort_index()
    coverage = length_counts.cumsum() / length_counts.sum()
    max_length = coverage[coverage >= 0.9].index[0]
    print("\nmax_length:",max_length)

    # padding
    corpus_pad_train = padding(corpus_train, max_length)
    corpus_pad_valid = padding(corpus_valid, max_length)
    corpus_pad_test = padding(corpus_test, max_length)



    # label 저장
    datasets = [(df_train, "train"), (df_valid, "valid"), (df_test, "test")]
    for data, name in datasets:
        save_path = PREPROCESSED_DIR / f"label_{name}.pkl"
        data = data["label"].to_numpy()

        with open(save_path, "wb") as f:
            pickle.dump(data, f)


    # word_to_id, id_to_word 저장
    if TOKENIZER != "subword":
        save_name = "word_to_id.pkl"
        save_path = PREPROCESSED_DIR / save_name
        with open(save_path, "wb") as f:
            pickle.dump(word_to_id, f)

        save_name = "id_to_word.pkl"
        save_path = PREPROCESSED_DIR / save_name
        with open(save_path, "wb") as f:
            pickle.dump(id_to_word, f)


    # corpus_pad 저장
    datasets = [(corpus_pad_train, "train"),
                (corpus_pad_valid, "valid"),
                (corpus_pad_test, "test")]
    for data, name in datasets:
        save_path = PREPROCESSED_DIR / f"corpus_pad_{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)