import torch
from transformers import BertForSequenceClassification
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)
import json

from dataloader import get_loader
from config import (
    BASE_DIR,
    CHECKPOINTS_DIR,
)


# dataset 불러오기
_, _, test_loader = get_loader()



## 모델 생성 ##

# 체크포인트 로드 위치
load_path = CHECKPOINTS_DIR / "best_bert.pt"
SAVE_DIR = BASE_DIR / "bert" / "results"

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 계산 장치 지정
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# 모델 생성
model = BertForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=2
).to(device)


# checkpoint 로드
model.load_state_dict(
    torch.load(load_path, map_location=device)
)

model.eval()



## 추론 ##

preds = []
targets = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Prediction"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds.extend(
            outputs.logits.argmax(dim=1).cpu().numpy()
        )
        targets.extend(
            batch['labels'].cpu().numpy()
        )


# acc/f1-score 출력
accuracy = sum(p == l for p, l in zip(preds, targets)) / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.4f}")

f1 = f1_score(targets, preds)
print(f"F1-score: {f1:.4f}")


# confusion matrix 출력
cm = confusion_matrix(targets, preds)
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues", values_format="d")

plt.savefig(SAVE_DIR/"confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# metrics 저장
metrics = {
    "test accuracy": accuracy,
    "f1-score": f1,
    "confusion matrix": cm.tolist()
}

with open(SAVE_DIR/"metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)