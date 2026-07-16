import pickle
from pathlib import Path
import sentencepiece as spm

from config import TOKENIZER

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# train, valid, test의 corpus와 label을 불러오기
def load_data():
    loaded = []

    for name in ["train", "valid", "test"]:
        load_path = DATA_DIR / f"preprocessed/corpus_pad_{name}.pkl"

        with open(load_path, "rb") as f:
            corpus = pickle.load(f)
        
        load_path = DATA_DIR / f"preprocessed/label_{name}.pkl"

        with open(load_path, "rb") as f:
            label = pickle.load(f)
        
        loaded.append((corpus, label))

    return loaded


# sentencepiece 불러오기
def load_sentencepiece():
    sp = spm.SentencePieceProcessor()

    sp.load(
        str(BASE_DIR/"checkpoints/tokenizer/sentencepiece.model")
    )

    return sp


# word_to_id 크기 불러오기
def load_vocab_size():
    if TOKENIZER != "subword":
        load_path = DATA_DIR / f"preprocessed/word_to_id.pkl"

        with open(load_path, "rb") as f:
            word_to_id = pickle.load(f)
        
        return len(word_to_id)
    else:
        sp = load_sentencepiece()

        return sp.get_piece_size()


# word_to_id, id_to_ word 불러오기
def load_vocab():
    loaded = []

    for name in ["word_to_id", "id_to_word"]:
        load_path = DATA_DIR / f"preprocessed/{name}.pkl"

        with open(load_path, "rb") as f:
            loaded.append(pickle.load(f))

    return loaded