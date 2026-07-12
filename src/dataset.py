import pickle
from pathlib import Path

MY_DIR = Path(__file__).resolve().parent
BASE_DIR = MY_DIR / "../data/preprocessed"

# train, valid, test의 corpus와 label을 불러오기
def load_data():
    loaded = []

    for name in ["train", "valid", "test"]:
        load_path = BASE_DIR / f"corpus_pad_{name}.pkl"

        with open(load_path, "rb") as f:
            corpus = pickle.load(f)
        
        load_path = BASE_DIR / f"label_{name}.pkl"

        with open(load_path, "rb") as f:
            label = pickle.load(f)
        
        loaded.append((corpus, label))

    return loaded


# word_to_id, id_to_ word 불러오기
def load_vocab():
    loaded = []

    for name in ["word_to_id", "id_to_word"]:
        load_path = BASE_DIR / f"{name}.pkl"

        with open(load_path, "rb") as f:
            loaded.append(pickle.load(f))

    return loaded