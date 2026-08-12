import torch
import torch.nn as nn
import torch.optim as optim

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        # x: batch, 100, 10

        x = x.permute(0, 2, 1)
        # x: batch, 10, 100

        x = self.cnn(x)
        # x: batch, 128, 25

        x = x.permute(0, 2, 1)
        # x: batch, 25, 128

        out, _ = self.lstm(x)
        # out: batch, 25, 256

        out = out[:, -1, :]
        # batch, 256

        out = self.fc(out)
        return out


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN_BiLSTM(23).to(device)
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)