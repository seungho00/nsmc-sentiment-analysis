from pathlib import Path
import sentencepiece as spm

from preprocess import load_raw_data, clean_data, split_train_valid


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data/raw/ratings_train.txt"
MODEL_PATH = BASE_DIR / "checkpoints/tokenizer"


# 데이터 불러오기
df, _ = load_raw_data()

# 데이터 정제
df = clean_data(df, "train")

# 검증 데이터 분리
df, _ = split_train_valid(df)


# 디렉토리 생성
MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# SentencePiece 학습용 텍스트 파일 저장
input_path = MODEL_PATH / "sp_train.txt"

df["document"].to_csv(
    input_path,
    index=False,
    header=False
)


# 학습 진행
spm.SentencePieceTrainer.train(
    input=str(input_path),
    model_prefix=str(MODEL_PATH / "sentencepiece"),
    vocab_size=8192,
    hard_vocab_limit=False,
    model_type="unigram",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=-1,
    eos_id=-1
)

print("SentencePiece 학습 완료")

# 학습용 파일 삭제
input_path.unlink()