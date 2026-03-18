import torch
from torch.utils.data import Dataset

class SeizureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
