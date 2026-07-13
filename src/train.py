import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import models
import dataset as my_dataset



## 데이터 불러오기 ##

(x_train, y_train), (x_valid, y_valid), _ = my_dataset.load_data()
word_to_id, _ = my_dataset.load_vocab()

# numpy에서 torch.tensor 타입으로 변환
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train, dtype=torch.float32)

x_valid = torch.tensor(x_valid)
y_valid = torch.tensor(y_valid, dtype=torch.float32)



## dataset 만들기 ##

# dataset 클래스 정의
class MovieDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# dataset, loader 만들기
dataset_train = MovieDataset(x_train, y_train)

loader_train = DataLoader(
    dataset_train,
    batch_size=32,
    shuffle=True
)

dataset_valid = MovieDataset(x_valid, y_valid)

loader_valid = DataLoader(
    dataset_valid,
    batch_size=32,
    shuffle=True
)



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


# 계산 장치 지정
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = model.to(device)



## 학습 진행 ##

# 체크포인트 저장 위치
SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"
save_path = SAVE_DIR / f"best_{model_type}.pt"

# 손실 함수, 옵티마이저 지정
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

epochs = 10

for epoch in range(epochs):

    # train
    model.train()

    total_loss = 0

    for inputs, labels in loader_train:
        
        # 장치 지정
        inputs = inputs.to(device)
        labels = labels.to(device)

        # gradient 초기화
        optimizer.zero_grad()

        # forward
        logits = model(inputs)

        # loss 계산
        loss = criterion(logits, labels)

        # backward
        loss.backward()

        # weight update
        optimizer.step()

        total_loss += loss.item()


    # validation
    model.eval()
    
    best_loss_valid = float("inf")
    total_loss_valid = 0

    with torch.no_grad():
        for inputs, labels in loader_valid:
            
            # 장치 지정
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward
            logits = model(inputs)

            # loss 계산
            loss = criterion(logits, labels)

            total_loss_valid += loss.item()


    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {total_loss / len(loader_train):.4f} | "
        f"Valid Loss: {total_loss_valid / len(loader_valid):.4f}"
    )


    # 체크포인트 저장
    if total_loss_valid < best_loss_valid:
        best_loss_valid = total_loss_valid

        torch.save(
            model.state_dict(),
            save_path
        )