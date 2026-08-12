import torch
from dataset import *
from cnn_bilstm_attention import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("./checkpoints/best_model.pth", map_location=device))
model.to(device)

model.eval()

correct = 0
total = 0

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