import torch
from dataset import *
from cnn_bilstm_attention import *
from sklearn.metrics import classification_report
import sys

# ===== Tee：同时输出到屏幕和 result.txt（追加模式） =====
class Tee:
    """同时输出到屏幕和 result.txt"""
    def __init__(self, file_path, mode="a", encoding="utf-8"):
        self.file = open(file_path, mode, encoding=encoding)

    def write(self, data):
        sys.__stdout__.write(data)   # 屏幕
        self.file.write(data)        # 文件
        self.file.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

sys.stdout = Tee("result.txt", mode="a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 3-seed 集成：加载 3 个模型 =====
model.load_state_dict(torch.load("./checkpoints/best_model.pth", map_location=device))
model.to(device)

model.eval()

correct = 0
total = 0

# ===== 集成评估（软投票：平均 3 个模型的 logits） =====
all_pred = []
all_label = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_pred.extend(pred.cpu().numpy())
        all_label.extend(y.cpu().numpy())

acc = correct / total
print(f"Test Accuracy: {acc*100:.2f}%")

print(classification_report(all_label, all_pred))