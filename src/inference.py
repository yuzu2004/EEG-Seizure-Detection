import torch
import numpy as np

def infer(model, loader, device):
    model.eval()
    probs = []

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            out = model(X)
            p = torch.softmax(out, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())

    return np.array(probs)
