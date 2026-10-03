"""One whole numeric scenario/record per token; fixed Fourier frequencies."""
import math

import torch
from torch import nn


class ScenarioTokens(nn.Module):
    def __init__(self, d_model=128, fourier_features=32, fourier_scales=(1, 2, 4, 8)):
        super().__init__()
        scales = torch.tensor([fourier_scales[i % len(fourier_scales)] for i in range(fourier_features)])
        self.register_buffer("frequencies", torch.randn(4, fourier_features) * scales)
        self.scenario = nn.Sequential(nn.Linear(4 + 2 * fourier_features + 3, d_model),
                                      nn.GELU(), nn.Linear(d_model, d_model))
        self.response = nn.Sequential(nn.Linear(1, d_model), nn.GELU())
        self.record_projection = nn.Linear(2 * d_model + 1, d_model)

    def derived_features(self, u):
        gap = 8 + 52 * u[:, 0:1]
        relative_speed = 25 - (15 + 10 * u[:, 1:2])
        duration, start = 1.5 + 1.5 * u[:, 2:3], .5 + 1.5 * u[:, 3:4]
        return torch.cat([relative_speed / 10, (gap - relative_speed * start) / 60,
                          (gap - relative_speed * (start + duration)) / 60], dim=1)

    def forward(self, x, z=None):
        phase = 2 * math.pi * (x @ self.frequencies)
        v = self.scenario(torch.cat([x, phase.sin(), phase.cos(), self.derived_features(x)], dim=1))
        observed = z is not None
        ez = self.response(z) if observed else torch.zeros_like(v)
        mask = torch.full_like(x[:, :1], float(observed))
        return self.record_projection(torch.cat([v, ez, mask], dim=1))
