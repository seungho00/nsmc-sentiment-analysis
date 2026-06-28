import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / '../data/ratings_train.txt'

print(pd.__version__)

try:
    df = pd.read_csv(file_path, sep='\t') # 탭으로 구분된 파일임을 명시
    print('데이터 로드 성공:')
    print(df.head())
except FileNotFoundError:
    print(f"오류: {file_path} 경로에서 파일을 찾을 수 없습니다. 파일 경로가 올바른지 확인해주세요.")
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")


# 결측 데이터 제거

# word_to_id 생성

# corpus 생성


# validation set 분리