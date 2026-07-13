import torch
from torch.utils.data import DataLoader
from pathlib import Path

import models
import dataset as my_dataset



## 데이터 로드 ##
_, _, (x_test, y_test) = my_dataset.load_data()
word_to_id, id_to_word = my_dataset.load_vocab()

# 데이터 타입 변환
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test, dtype=torch.float32)



## dataset 만들기 ##

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


dataset_test = MovieDataset(x_test, y_test)

loader_test = DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False
)



## 계산 장치 지정 ##
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")



## 하이퍼파라미터 설정 ##
vocab_size = len(word_to_id)
embedding_dim = 128
hidden_size = 64
learning_rate = 0.001



## 모델 생성 ##

model_type = "LSTM"

if model_type == "LSTM":
    model = models.lstm.SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size
    )

# 체크포인트 로드 위치
LOAD_DIR = Path(__file__).resolve().parent / "checkpoints"
load_path = LOAD_DIR / f"best_{model_type}.pt"

# checkpoint 로드
model.load_state_dict(
    torch.load(load_path)
)
model = model.to(device)

model.eval()



## 추론 ##

correct = 0
num_samples = 0

with torch.no_grad():

    for inputs, labels in loader_test:

        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)

        preds = (torch.sigmoid(logits) >= 0.5).float()

        correct += (preds == labels).sum().item()
        num_samples += len(labels)


accuracy = correct / num_samples

print(f"Test Accuracy: {accuracy:.4f}")