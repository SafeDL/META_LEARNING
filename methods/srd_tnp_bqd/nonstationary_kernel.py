"""Independent physical plus response-gated multiscale Matérn-5/2 kernel."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def inv_softplus(value):
    return math.log(math.expm1(value))


def matern52(distance, lengthscale):
    r = math.sqrt(5) * distance / lengthscale
    return (1 + r + r.square() / 3) * torch.exp(-r)


class ResidualKernel(nn.Module):
    def __init__(self, config, h_dim=32, mode="nonstationary"):
        super().__init__()
        self.mode, self.local_lower = mode, config["sigma_local_lower_bound"]
        self.gate = nn.Sequential(nn.Linear(4 + h_dim, config["gate_hidden"]), nn.Tanh(), nn.Linear(config["gate_hidden"], 3))
        self.raw_local_sigma = nn.Parameter(torch.tensor(inv_softplus(config["sigma_local"] - self.local_lower)))
        self.raw_shared_sigma = nn.Parameter(torch.tensor(inv_softplus(config["sigma_shared"])))
        self.raw_local_length = nn.Parameter(torch.tensor(inv_softplus(config["lengthscale_local"])))
        self.raw_lengths = nn.Parameter(torch.tensor([inv_softplus(v) for v in config["lengthscales"]]))
        self.raw_h_length = nn.Parameter(torch.tensor(inv_softplus(config["lengthscale_h"])))
        self.double()

    @property
    def sigma_local(self):
        return self.local_lower + F.softplus(self.raw_local_sigma)

    def tensors(self, x, h):
        p = next(self.parameters())
        return (torch.as_tensor(x, dtype=torch.float64, device=p.device),
                torch.as_tensor(h, dtype=torch.float64, device=p.device))

    def scale_weights(self, x, h):
        x, h = self.tensors(x, h)
        a = F.softplus(self.gate(torch.cat([x, h], -1)))
        return a / (torch.linalg.vector_norm(a, dim=-1, keepdim=True) + 1e-12)

    def matrix(self, x1, h1, x2, h2):
        x1, h1 = self.tensors(x1, h1)
        x2, h2 = self.tensors(x2, h2)
        distances = torch.cdist(x1, x2)
        local = self.sigma_local.square() * matern52(distances, F.softplus(self.raw_local_length))
        if self.mode == "stationary":
            # One physical Matérn, with matched marginal prior variance.
            return (self.sigma_local.square() + F.softplus(self.raw_shared_sigma).square()) * matern52(distances, F.softplus(self.raw_local_length))
        w1, w2 = self.scale_weights(x1, h1), self.scale_weights(x2, h2)
        mixture = torch.zeros_like(distances)
        for i, length in enumerate(F.softplus(self.raw_lengths)):
            mixture = mixture + w1[:, i:i + 1] * w2[:, i:i + 1].T * matern52(distances, length)
        kh = torch.exp(-torch.cdist(h1, h2).square() / (2 * F.softplus(self.raw_h_length).square()))
        return local + F.softplus(self.raw_shared_sigma).square() * kh * mixture

    def diagonal(self, x, h):
        x, h = self.tensors(x, h)
        if self.mode == "stationary":
            return (self.sigma_local.square() + F.softplus(self.raw_shared_sigma).square()).expand(len(x))
        return self.sigma_local.square() + F.softplus(self.raw_shared_sigma).square() * self.scale_weights(x, h).square().sum(-1)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self
