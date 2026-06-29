import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / '../data/ratings_train.txt'

# 데이터 불러오기

try:
    df = pd.read_csv(file_path, sep='\t') # 탭으로 구분된 파일임을 명시
    print('데이터 로드 성공')
    # print(df.head())
except FileNotFoundError:
    print(f"오류: {file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise  # 예외를 다시 발생시키는 명령어, 호출한 쪽에서 예외를 받아 다른 처리를 할 수도 있다


## 결측 데이터 제거 ##

# 결측값 개수
print("\n결측값의 수:", df["document"].isna().sum())

# 결측 데이터 출력
print("결측 데이터 목록")
print(df[df["document"].isna()])

# 빈 문자열 확인
print("\n빈 문자열 데이터 목록")
print(df[df["document"] == ""])

# 공백만 있는 문자열
print("\n공백만 있는 문자열 데이터 목록")
print(df[df["document"].str.strip() == ""])

# 결측 데이터 제거
print("\n삭제 전:", len(df))

df = df.dropna(subset=["document"])

print("삭제 후:", len(df))
print("결측값의 수:", df["document"].isna().sum())



## word_to_id 생성 ##

# 특수 문자도 따로 처리 될 수 있도록 공백 추가
df["document"] = df["document"].str.replace(
    r"([.,!?])", r" \1 ", regex=True
)

tokenized_documents = df["document"].str.split()
word_to_id = {}
id_to_word = {}
for words in tokenized_documents:
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

for i in range(15):
    print(i,':', id_to_word[i])





# corpus 생성


# validation set 분리