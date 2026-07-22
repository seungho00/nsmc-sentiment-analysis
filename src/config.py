# ===== Data =====
VALID_SIZE = 0.1
RANDOM_STATE = 42

# ===== Preprocessing =====
TOKENIZER = "morpheme"
MIN_FREQ = 3
PAD_ID = 0
UNK_ID = 1

# ===== Model =====
MODEL_TYPE = "GRU"      # RNN, LSTM, GRU, BiLSTM, BiGRU
DEVICE = "cpu"          # None이면 자동 선택
EMBEDDING_DIM = 32
HIDDEN_SIZE = 32
DROPOUT = 0.2

# ===== Training =====
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10

# ===== Experiment =====
EXPERIMENT_NAME = "experiment3"