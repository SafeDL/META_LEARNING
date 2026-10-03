"""Small shared scene encoder with one historical risk head per source."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ResponseEncoder(nn.Module):
    def __init__(self, source_count: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32),
        )
        self.heads = nn.Linear(32, source_count)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.heads(self.encoder(coordinates))


def predict_response(model: ResponseEncoder, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model(torch.as_tensor(x, dtype=torch.float32,
                                           device=device)).sigmoid()
    return prediction.cpu().numpy()
