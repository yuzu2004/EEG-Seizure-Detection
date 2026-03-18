import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.dataset import SeizureDataset
from src.model import SpikeNet
from src.train import train

CSV_PATH = "data/seizure_event_candidates.csv"

FEATURE_COLS = [...]  # giữ nguyên list của bạn

def main():
    df = pd.read_csv(CSV_PATH)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y = df['label'].values

    dataset = SeizureDataset(X, y)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SpikeNet(len(FEATURE_COLS)).to(device)

    train(model, loader, device)

if __name__ == "__main__":
    main()
