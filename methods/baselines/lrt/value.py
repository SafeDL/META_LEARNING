"""Permutation-invariant prediction of frozen CG-BRT continuation returns."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import NOMINAL
from .fantasy import acquisition_values


def mlp(inputs, width, output):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, output))


def make_features(
    models,
    remaining,
    candidates,
    reference_probability,
    trust,
    horizon,
    step,
    losses,
    seed,
    *,
    draws=None
):
    available = np.flatnonzero(remaining)
    probability = np.stack([m.probabilities()[available] for m in models])
    order = np.argsort(-probability, axis=-1, kind="stable")
    boundary = probability[
        np.arange(len(models)), order[:, min(horizon, len(available)) - 1]
    ]
    frontier = np.argsort(abs(probability - boundary[:, None]), axis=-1, kind="stable")[
        :, :8
    ]
    values = np.stack(
        [
            acquisition_values(
                m,
                candidates,
                remaining,
                horizon,
                seed,
                draws=None if draws is None else draws[i],
            )
            for i, m in enumerate(models)
        ]
    )
    extra = np.empty((len(models), len(candidates), 8))
    extra[..., 0] = values / horizon
    extra[..., 1] = losses
    extra[..., 2] = horizon / len(available)
    extra[..., 3] = horizon / 200
    extra[..., 4] = step / 200
    extra[..., 5] = trust
    extra[..., 6] = np.array([m.kappa / 2 for m in models])[:, None]
    extra[..., 7] = np.array([m.target_probability for m in models])[:, None]
    return {
        "source": np.stack([m.source_tokens(available) for m in models]).astype(
            np.float32
        ),
        "x": models[0].prior.x[available].astype(np.float32),
        "q": probability.astype(np.float32),
        "reference": reference_probability[available].astype(np.float32),
        "candidates": np.searchsorted(available, candidates),
        "top": order[:, : min(horizon, len(available))],
        "frontier": frontier,
        "extra": extra.astype(np.float32),
        "horizon": horizon,
        "lookahead": values,
    }


class ContinuationValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = mlp(4, 32, 32)
        self.point = mlp(39, 64, 64)
        self.head = nn.Sequential(
            nn.Linear(264, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        device = next(self.parameters()).device

        def tensor(name, dtype=torch.float32):
            return torch.as_tensor(features[name], dtype=dtype, device=device)

        q = tensor("q")
        candidates = tensor("candidates", torch.long)
        horizon = features["horizon"]
        if horizon == 1:
            return q[:, candidates]
        source = tensor("source")
        embedding = (self.source(source) * source[..., 2, None]).sum(-2)
        inputs = torch.cat(
            (
                embedding,
                tensor("x")[None].expand(len(q), -1, -1),
                q[NOMINAL : NOMINAL + 1, :, None].expand(len(q), -1, -1),
                tensor("reference")[None, :, None].expand(len(q), -1, -1),
            ),
            -1,
        )
        point = self.point(inputs)
        models = torch.arange(len(q), device=device)[:, None]
        context = torch.cat(
            (
                point.mean(1),
                point[models, tensor("top", torch.long)].mean(1),
                point[models, tensor("frontier", torch.long)].mean(1),
            ),
            -1,
        )
        head = torch.cat(
            (
                point[:, candidates],
                context[:, None].expand(-1, len(candidates), -1),
                tensor("extra"),
            ),
            -1,
        )
        return q[:, candidates] + (horizon - 1) * self.head(head).squeeze(-1).sigmoid()


def value_loss(predicted, truth, horizon):
    truth = torch.as_tensor(truth, device=predicted.device, dtype=predicted.dtype)
    mse = ((predicted / horizon - truth / horizon) ** 2).mean()
    difference = truth[:, None] - truth[None]
    first, second = torch.where(difference > 0)
    rank = predicted.sum() * 0
    if len(first):
        rank = F.softplus(
            -(predicted[:, first] - predicted[:, second]) / (0.05 * horizon)
        ).mean()
    return mse + 0.2 * rank
