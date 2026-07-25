from pathlib import Path

# ===== Data =====
VALID_SIZE = 0.1
RANDOM_STATE = 42
TOKENIZER = "klue/bert-base"

# ===== Training =====
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10

# ===== Path =====
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = BASE_DIR / "bert/preprocessed"