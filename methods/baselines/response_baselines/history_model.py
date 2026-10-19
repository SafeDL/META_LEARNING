"""Frozen six-source neural inference."""
import numpy as np
import torch
from torch import nn

from .config import HISTORY_MODELS


class HistoryResponseModel(nn.Module):

    def __init__(self, source_count=6):
        super().__init__()
        self.source_count = source_count
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * source_count),
        )

    def forward(self, x):
        output = self.layers(x)
        return output[:, :self.source_count].sigmoid(), output[:, self.
                                                               source_count:]


def predict(x, seed):
    state = torch.load(HISTORY_MODELS / f"emulator_{seed}.pt",
                       weights_only=True)
    risks = np.zeros((len(x), 6))
    collision_scores = np.zeros((len(x), 6))
    for family in (0, 1):
        indices = np.flatnonzero(x[:, 4] == family)
        model = HistoryResponseModel().cuda().eval().requires_grad_(False)
        model.load_state_dict(state["states"][str(family)])
        with torch.no_grad():
            risk, logits = model(
                torch.as_tensor(
                    x[indices, :4],
                    dtype=torch.float32,
                    device="cuda",
                ))
        risks[indices] = risk.cpu().numpy()
        collision_scores[indices] = logits.sigmoid().cpu().numpy()
    return risks, collision_scores
