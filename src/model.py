import torch
import torch.nn as nn
import numpy as np

# ===== Spike function =====
class SurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        surrogate = 2.5 / (1.0 + (np.pi * x * 1.5) ** 2)
        return grad_output * surrogate

spike_fn = SurrGradSpike.apply


# ===== LIF =====
class LIF(nn.Module):
    def __init__(self, tau=20.0, thr=0.65):
        super().__init__()
        self.tau = tau
        self.thr = thr
        self.bias = nn.Parameter(torch.tensor(0.1))

    def forward(self, I, V, epoch=0, total_epochs=10):
        anneal = 1 - 0.25 * (epoch / total_epochs)
        thr = self.thr * anneal

        V = V + (I - V) / self.tau + self.bias
        spk = spike_fn(V - thr)
        V = V - spk * thr
        return spk, V


# ===== Encoder =====
def gaussian_spike_encoder(x, timesteps=20, sigma=0.65):
    B, F = x.shape
    device = x.device
    centers = torch.linspace(-2, 2, timesteps, device=device)

    spikes = torch.zeros(B, timesteps, F, device=device)

    for t in range(timesteps):
        prob = torch.exp(-(x - centers[t])**2 / (2 * sigma**2))
        spikes[:, t] = (torch.rand_like(prob) < prob).float()

    return spikes


# ===== Model =====
class SpikeNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.lif1 = LIF(28.0, 0.1)
        self.lif2 = LIF(14.0, 0.35)
        self.lif3 = LIF(16.0, 0.5)

        self.scale1 = 35.0
        self.scale2 = 18.0
        self.scale3 = 10.1

    def forward(self, x, epoch=0, total_epochs=10):
        B = x.size(0)
        x = gaussian_spike_encoder(x)

        V1 = torch.zeros(B, 128, device=x.device)
        V2 = torch.zeros(B, 64, device=x.device)
        V3 = torch.zeros(B, 2, device=x.device)

        spk_sum = torch.zeros(B, 2, device=x.device)

        for t in range(x.size(1)):
            s1, V1 = self.lif1(self.fc1(x[:, t]) * self.scale1, V1, epoch, total_epochs)
            s2, V2 = self.lif2(self.fc2(s1) * self.scale2, V2, epoch, total_epochs)
            s3, V3 = self.lif3(self.fc3(s2) * self.scale3, V3, epoch, total_epochs)
            spk_sum += s3

        return V3, spk_sum.mean()
