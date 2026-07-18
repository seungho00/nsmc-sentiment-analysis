from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from config import MODEL_TYPE

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = BASE_DIR / f"results/{MODEL_TYPE.lower()}"

# pickle 불러오기
load_path = HISTORY_DIR / "history.pkl"
with open(load_path, "rb") as f:
    history = pickle.load(f)

# 그래프 저장위치 지정
loss_save_path = HISTORY_DIR / "loss.png"
acc_save_path = HISTORY_DIR / "accuracy.png"


# 에포크 크기 계산
epochs = range(1, len(history["loss_train"]) + 1)

# Loss
plt.figure(figsize=(6, 4))
plt.plot(epochs, history["loss_train"], label="Train")
plt.plot(epochs, history["loss_valid"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid(True)
plt.savefig(loss_save_path, dpi=300, bbox_inches="tight")


# Accuracy
plt.figure(figsize=(6, 4))
plt.plot(epochs, history["acc_train"], label="Train")
plt.plot(epochs, history["acc_valid"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(acc_save_path, dpi=300, bbox_inches="tight")