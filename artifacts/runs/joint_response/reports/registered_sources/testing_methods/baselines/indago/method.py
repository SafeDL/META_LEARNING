"""Indago's two-class MLP trained from allowed historical response archives."""

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from testing_methods.common.response import archive_fingerprints


@dataclass(frozen=True)
class TrainingConfig:
    layers: int = 4
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.0003
    validation_fraction: float = 0.2
    seed: int = 1409


DEFAULT = TrainingConfig()


class FailurePredictor(nn.Module):
    def __init__(self, dimension, layers=4):
        super().__init__()
        self.dimension, self.layers = dimension, layers
        blocks = []
        for i in range(layers):
            blocks.extend(
                [nn.Linear(dimension if i == 0 else 32, 32), nn.BatchNorm1d(32), nn.ReLU()]
            )
            if layers > 1:
                blocks.append(nn.Dropout(0.5))
        blocks.append(nn.Linear(32, 2))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(torch.as_tensor(x, dtype=torch.float32))

    def probabilities(self, x):
        return self(x).softmax(-1)[:, 1]


def train(history, destination, *, held_family=None, config=DEFAULT):
    history = tuple(history)
    if not history or any(a.role != "history" for a in history):
        raise ValueError("Indago training receives allowed historical objects only")
    if held_family is not None and any(a.family == held_family for a in history):
        raise ValueError("Held family entered Indago classifier training")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    x = np.concatenate([a.x for a in history])
    labels = np.concatenate([a.collision for a in history]).astype(int)
    # Matching scenarios across sources stay together in the validation partition.
    unique, groups = np.unique(x, axis=0, return_inverse=True)
    order = generator.permutation(len(unique))
    count = max(1, round(config.validation_fraction * len(unique)))
    validation = np.isin(groups, order[:count])
    training_indices = np.flatnonzero(~validation)
    counts = np.bincount(labels[training_indices], minlength=2)
    weights = torch.tensor(
        np.divide(
            len(training_indices),
            np.count_nonzero(counts) * counts,
            out=np.zeros(2, dtype=float),
            where=counts > 0,
        ),
        dtype=torch.float32,
    )
    model = FailurePredictor(x.shape[1], layers=config.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    y = torch.as_tensor(labels, dtype=torch.long)
    best, best_state, curve = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        indices = generator.permutation(training_indices)
        for batch in np.array_split(
            indices, max(1, int(np.ceil(len(indices) / config.batch_size)))
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[batch]), y[batch], weight=weights)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(F.cross_entropy(model(x[validation]), y[validation], weight=weights))
        curve.append({"epoch": epoch, "historical_validation_cross_entropy": loss})
        if loss < best:
            best, best_state = loss, copy.deepcopy(model.state_dict())
    checkpoint = {
        "format": "indago-historical-classifier",
        "dimension": x.shape[1],
        "config": asdict(config),
        "state_dict": best_state,
        "history_ids": [a.source_id for a in history],
        "history_fingerprints": archive_fingerprints(history),
        "held_family": held_family,
        "data_role": "finished historical systems; not target policy training logs",
        "training_rows": len(training_indices),
        "validation_rows": int(validation.sum()),
        "class_weights": weights.tolist(),
        "curve": curve,
        "historical_training_class_counts": counts.tolist(),
        "missing_class_policy": "balanced weights over observed classes; retain two logits and train the observed evidence",
        "training_s": time.perf_counter() - started,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return {k: v for k, v in checkpoint.items() if k != "state_dict"}


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["format"] != "indago-historical-classifier":
        raise ValueError("Indago needs its historical two-class classifier")
    model = FailurePredictor(checkpoint["dimension"], checkpoint["config"]["layers"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False), checkpoint
