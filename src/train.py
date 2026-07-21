import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle

import models
import dataset as my_dataset
from config import (
    MODEL_TYPE,
    DEVICE,
    EMBEDDING_DIM,
    HIDDEN_SIZE,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS
    )



## 데이터 불러오기 ##

(x_train, y_train), (x_valid, y_valid), _ = my_dataset.load_data()
vocab_size = my_dataset.load_vocab_size()

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
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataset_valid = MovieDataset(x_valid, y_valid)

loader_valid = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE,
    shuffle=False
)



## 모델 생성 ##

MODEL_TYPES = {
    "RNN": models.rnn.SentimentRNN,
    "LSTM": models.lstm.SentimentLSTM,
    "GRU": models.gru.SentimentGRU,
    "BiLSTM": models.bilstm.SentimentBiLSTM,
}
model_constructor = MODEL_TYPES[MODEL_TYPE]

model = model_constructor(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE
)


# 계산 장치 지정
if DEVICE is not None:
    device = torch.device(DEVICE)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)

print(f"Using device: {device}")



## 학습 진행 ##

# 손실 함수, 옵티마이저 지정
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 체크포인트 저장 위치
BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_save_path = CHECKPOINT_DIR / f"best_{MODEL_TYPE}.pt"


# 학습 과정 저장용 변수
losses_train = []
losses_valid = []

accs_train = []
accs_valid = []

# 검증 데이터에 대한 최소 loss 저장 변수
best_loss_valid = float("inf")

for epoch in range(EPOCHS):

    # train
    model.train()

    total_loss_train = 0
    correct_train = 0

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

        total_loss_train += loss.item()

        # 정확도 측정
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).long()
            correct_train += (pred.squeeze() == labels).sum().item()

    # 현재 에포크의 손실, 정확도 저장
    losses_train.append(total_loss_train / len(loader_train))
    accs_train.append(correct_train / len(loader_train.dataset))



    ## validation ##

    model.eval()
    
    total_loss_valid = 0
    valid_correct = 0

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
            
            # valid 정확도 측정
            pred = (torch.sigmoid(logits) > 0.5).long()
            valid_correct += (pred.squeeze() == labels).sum().item()

        # 현재 에포크의 손실, 정확도 저장
        losses_valid.append(total_loss_valid / len(loader_valid))
        accs_valid.append(valid_correct / len(loader_valid.dataset))


    # 학습 과정 출력
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {total_loss_train / len(loader_train):.4f} | "
        f"Valid Loss: {total_loss_valid / len(loader_valid):.4f}"
    )


    # 체크포인트 저장
    if total_loss_valid < best_loss_valid:
        best_loss_valid = total_loss_valid

        torch.save(
            model.state_dict(),
            model_save_path
        )


# 학습 과정 저장
HISTORY_DIR = BASE_DIR / "results" / MODEL_TYPE.lower()

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

history = {
    "loss_train": losses_train,
    "acc_train": accs_train,
    "loss_valid": losses_valid,
    "acc_valid": accs_valid
}
history_save_path = HISTORY_DIR / "history.pkl"
with open(history_save_path, "wb") as f:
    pickle.dump(history, f)