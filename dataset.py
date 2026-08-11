import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


path = r"D:\Python Projects\20260810EMG_CNN_LSTM\data\raw\Ninapro_DB1.csv"

df = pd.read_csv(path)

df_action = df[df["stimulus"] != 0]
emg_cols = [f"emg_{i}" for i in range(10)]

X = df_action[emg_cols].values
y = df_action["stimulus"].values
subject = df_action["subject"].values

train_subjects = list(range(1, 21))
val_subjects = [21, 22, 23]
test_subjects = [24, 25, 26, 27]

train_mask = np.isin(subject, train_subjects)
val_mask = np.isin(subject, val_subjects)
test_mask = np.isin(subject, test_subjects)

X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[val_mask]
y_val = y[val_mask]
X_test = X[test_mask]
y_test = y[test_mask]

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def create_windows(
        X,
        y,
        window_size=100,
        stride=50
):

    X_windows=[]
    y_windows=[]


    for start in range(
        0,
        len(X)-window_size,
        stride
    ):

        end=start+window_size


        # 当前窗口
        window=X[start:end]


        # 当前窗口标签
        labels=y[start:end]


        # 多数投票
        label=np.bincount(labels).argmax()


        X_windows.append(window)
        y_windows.append(label)


    X_windows=np.array(X_windows)
    y_windows=np.array(y_windows)


    return X_windows,y_windows

X_train_win,y_train_win=create_windows(
    X_train,
    y_train,
    window_size=100,
    stride=50
)
X_val_win,y_val_win=create_windows(
    X_val,
    y_val,
    window_size=100,
    stride=50
)

X_test_win,y_test_win=create_windows(
    X_test,
    y_test,
    window_size=100,
    stride=50
)
y_train_win = y_train_win - 1
y_val_win = y_val_win - 1
y_test_win = y_test_win - 1

class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


train_dataset = EMGDataset(
    X_train_win,
    y_train_win
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = EMGDataset(
    X_val_win,
    y_val_win
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataset = EMGDataset(
    X_test_win,
    y_test_win
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)