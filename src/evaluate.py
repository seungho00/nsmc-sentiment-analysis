import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

import models
import dataset as my_dataset
from config import (
    MODEL_TYPE,
    DEVICE,
    EMBEDDING_DIM,
    HIDDEN_SIZE,
    BATCH_SIZE,
)


## 데이터 로드 ##
_, _, (x_test, y_test) = my_dataset.load_data()
vocab_size = my_dataset.load_vocab_size()

# 데이터 타입 변환
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)



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
    batch_size=BATCH_SIZE,
    shuffle=False
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

print(f"Using device: {device}")



## 모델 생성 ##

# 체크포인트 로드 위치
BASE_DIR = Path(__file__).resolve().parent.parent
LOAD_DIR = BASE_DIR / "checkpoints"
load_path = LOAD_DIR / f"best_{MODEL_TYPE}.pt"
save_path = BASE_DIR / f"results/{MODEL_TYPE}/confusion_matrix.png"

# 모델 타입 선택
MODEL_TYPES = {
    "RNN": models.rnn.SentimentRNN,
    "LSTM": models.lstm.SentimentLSTM
}
model_constructor = MODEL_TYPES[MODEL_TYPE]

model = model_constructor(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE
)

# checkpoint 로드
model.load_state_dict(
    torch.load(load_path, map_location=device)
)
model = model.to(device)

model.eval()



## 추론 ##

preds = []
labels = []

with torch.no_grad():

    for inputs, targets in loader_test:

        inputs = inputs.to(device)

        logits = model(inputs)

        pred = (torch.sigmoid(logits) >= 0.5).long().cpu()

        preds.extend(pred.numpy())
        labels.extend(targets.numpy())


accuracy = sum(p == l for p, l in zip(preds, labels)) / len(loader_test.dataset)

print(f"Test Accuracy: {accuracy:.4f}")


# confusion matrix
cm = confusion_matrix(labels, preds)
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()


# f1-score
f1 = f1_score(labels, preds)
print(f"F1-score: {f1:.4f}")