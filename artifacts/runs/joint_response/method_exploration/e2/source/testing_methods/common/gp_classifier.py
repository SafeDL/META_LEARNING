"""Shared Matérn kernel and optional feature network for joint response models."""

import math

import torch
from torch import nn


class ClassificationKernel(nn.Module):
    def __init__(self, dimension, *, deep=False):
        super().__init__()
        self.dimension = dimension
        self.deep = deep
        if deep:
            layers = []
            for incoming, outgoing in zip((dimension, 32, 32, 32), (32, 32, 32, 32)):
                if layers:
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(incoming, outgoing))
            self.embedding = nn.Sequential(*layers)
        else:
            self.embedding = nn.Identity()
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(0.3), dtype=torch.float64))
        self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.mean = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def features(self, x):
        x = torch.as_tensor(x, device=self.mean.device, dtype=torch.float64)
        if self.deep:
            return self.embedding(x.to(next(self.embedding.parameters()).dtype)).double()
        return x

    def covariance(self, x, other=None):
        z = self.features(x)
        z_other = z if other is None else self.features(other)
        distance = torch.cdist(z, z_other) / self.log_lengthscale.exp().clamp_min(0.01)
        r = math.sqrt(5) * distance
        return self.log_scale.exp().square() * (1 + r + r.square() / 3) * (-r).exp()
