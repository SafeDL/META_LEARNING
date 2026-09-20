"""Masked trajectory reconstruction encoder for collision post-analysis."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MaskedTrajectoryEncoder(nn.Module):
    def __init__(self, feature_dim: int = 8, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.decoder = nn.Linear(hidden, feature_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(self.input(x), src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (latent * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.decoder(latent), pooled


def prepare_trajectories(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            values = data["values"]
        # Common ego-origin frame; both vehicles retain their interaction offset.
        origin = values[0, 1:3].copy()
        sequence = np.column_stack((
            values[:, 1] - origin[0], values[:, 2] - origin[1], values[:, 3], values[:, 5],
            values[:, 7] - origin[0], values[:, 8] - origin[1], values[:, 9], values[:, 11],
        )).astype(np.float32)
        sequences.append(sequence)
    length = max(len(sequence) for sequence in sequences)
    batch = np.zeros((len(sequences), length, 8), dtype=np.float32)
    padding = np.ones((len(sequences), length), dtype=bool)
    for i, sequence in enumerate(sequences):
        batch[i, :len(sequence)] = sequence
        padding[i, :len(sequence)] = False
    scale = np.std(np.concatenate(sequences), axis=0)
    scale[scale < 1e-6] = 1.0
    batch /= scale
    return batch, padding


def learn_embeddings(paths: list[str], seed: int, epochs: int = 80) -> np.ndarray:
    torch.manual_seed(seed)
    values, padding = prepare_trajectories(paths)
    x = torch.as_tensor(values)
    pad = torch.as_tensor(padding)
    model = MaskedTrajectoryEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        real = ~padding
        masked = np.zeros_like(real)
        for i in range(len(values)):
            indices = np.flatnonzero(real[i])
            if len(indices):
                start = int(rng.choice(indices))
                masked[i, start:min(start + max(2, len(indices) // 10), len(indices))] = True
        masked_x = x.clone(); masked_x[torch.as_tensor(masked)] = 0.0
        reconstruction, _ = model(masked_x, pad)
        mask_tensor = torch.as_tensor(masked).unsqueeze(-1).expand_as(reconstruction)
        loss = ((reconstruction - x) ** 2)[mask_tensor].mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        _, embeddings = model(x, pad)
    return embeddings.numpy()

