import torch
import time
from dataset import *
from cnn_bilstm_attention import *
import torch.optim as optim
import random
import sys

# 训练种子：python train.py 42 / 123 / 2026
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print(f"Seed: {seed}")

class Tee:
    """同时输出到屏幕和 result.txt"""

    def __init__(self, file_path, mode="w", encoding="utf-8"):
        self.file = open(file_path, mode, encoding=encoding)

    def write(self, data):
        sys.__stdout__.write(data)  # 屏幕
        self.file.write(data)  # 文件
        self.file.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


sys.stdout = Tee("result.txt", mode="w")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    loss = total_loss / len(loader)
    acc = correct / total
    return (loss, acc)

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    loss = total_loss / len(loader)
    acc = correct / total
    return (loss, acc)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='max', factor=0.5, patience=3
)

epochs = 20
best_acc = 0
patience = 5
wait = 0
total_start_time = time.time()

for epoch in range(epochs):
    print(f"\n{'=' * 60}")
    print(f"🔄 Epoch [{epoch + 1}/{epochs}] 开始训练...")
    print(f"{'=' * 60}")
    epoch_start_time = time.time()
    print(f"📊 训练阶段: 正在学习...")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"📊 验证阶段: 正在评估...")
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start_time
    avg_epoch_time = (time.time() - total_start_time) / (epoch + 1)
    remain_time = avg_epoch_time * (epochs - epoch - 1)

    scheduler.step(val_acc)

    print(
        f"Epoch:{epoch + 1}/{epochs}, "
        f"Train Loss:{train_loss:.4f}, "
        f"Train Acc:{train_acc:.4f}, "
        f"Val Loss:{val_loss:.4f}, "
        f"Val Acc:{val_acc:.4f}, "
        f"LR:{optimizer.param_groups[0]['lr']:.1e}, "
        f"Time:{epoch_time:.2f}s, "
        f"ETA:{remain_time / 60:.2f}min"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        wait = 0
        torch.save(model.state_dict(),f"checkpoints/best_model.pth")
        print("Best model saved!")
    else:
        wait = wait + 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}(patience={patience})")
            break

total_time = time.time() - total_start_time
print(f"Training finished! Total training time: {total_time/60:.2f} minutes")

# ========== 训练完成提示 ==========
print("\n" + "=" * 60)
print("🎉 训练完成!")
print("=" * 60)
print(f"📊 最佳验证准确率: {best_acc:.4%}")
print(f"⏱️  总训练时间: {total_time / 60:.2f} 分钟")
print(f"💾 模型保存在: checkpoints/best_model.pth")
print("=" * 60)