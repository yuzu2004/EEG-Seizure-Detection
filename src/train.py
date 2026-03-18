import torch
import torch.nn as nn
import torch.optim as optim

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        ce = nn.functional.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()


def train(model, loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = FocalLoss()

    model.train()
    for epoch in range(epochs):
        total = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits, reg = model(X, epoch, epochs)

            loss = criterion(logits, y) + reg
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {epoch+1}: loss={total:.4f}")
