import torch
import time
from dataset import *
from cnn_bilstm_attention import *
import torch.optim as optim


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


epochs = 5
best_acc = 0
total_start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start_time
    avg_epoch_time = (time.time() - total_start_time) / (epoch + 1)
    remain_time = avg_epoch_time * (epochs - epoch - 1)

    print(
        f"Epoch:{epoch + 1}/{epochs}, "
        f"Train Loss:{train_loss:.4f}, "
        f"Train Acc:{train_acc:.4f}, "
        f"Val Loss:{val_loss:.4f}, "
        f"Val Acc:{val_acc:.4f}, "
        f"Time:{epoch_time:.2f}s, "
        f"ETA:{remain_time / 60:.2f}min"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),"checkpoints/best_model.pth")
        print("Best model saved!")

total_time = time.time() - total_start_time
print(f"Training finished! Total training time: {total_time/60:.2f} minutes")