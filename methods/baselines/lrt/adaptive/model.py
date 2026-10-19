"""A frozen set representation and zero-inflated beta-binomial count head."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import NOMINAL
from methods.common.counts import count_log_prob, expected_count


class RemainingDiscoveries(nn.Module):
    def __init__(self, value_model):
        super().__init__()
        self.source = copy.deepcopy(value_model.source).requires_grad_(False)
        self.point = copy.deepcopy(value_model.point).requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(136, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    @torch.no_grad()
    def features(self, models, remaining, anchor, queries, evidence, budget, lengths):
        device = next(self.parameters()).device
        available = np.flatnonzero(remaining)
        q = np.stack([m.probabilities()[available] for m in models])
        source = torch.as_tensor(
            np.stack([m.source_tokens(available) for m in models]),
            dtype=torch.float32,
            device=device,
        )
        embedded = (self.source(source) * source[..., 2, None]).sum(-2)
        x = torch.as_tensor(
            models[0].prior.x[available], dtype=torch.float32, device=device
        )
        nominal = torch.as_tensor(q[NOMINAL], dtype=torch.float32, device=device)
        reference = torch.as_tensor(
            np.exp(anchor.gated[0])[available], dtype=torch.float32, device=device
        )
        points = self.point(
            torch.cat(
                (
                    embedded,
                    x[None].expand(len(models), -1, -1),
                    nominal[None, :, None].expand(len(models), -1, -1),
                    reference[None, :, None].expand(len(models), -1, -1),
                ),
                -1,
            )
        )
        found = sum(row["risk"] > 0.5 for row in queries)
        brier = (
            np.mean(
                [
                    (float(row["risk"] > 0.5) - ev["prequery_probability"]) ** 2
                    for row, ev in zip(queries[-10:], evidence[-10:])
                ]
            )
            if queries
            else 0.0
        )
        weights = source[:, 0, :, 2].cpu().numpy()
        entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(-1) / np.log(
            weights.shape[1]
        )
        order = np.argsort(-q, axis=-1, kind="stable")
        result = []
        for b in lengths:
            top = torch.as_tensor(order[:, : min(b, len(available))], device=device)
            pooled = points[
                torch.arange(len(models), device=device)[:, None], top
            ].mean(1)
            extra = np.column_stack(
                (
                    np.full(len(models), found / budget),
                    np.full(len(models), len(queries) / budget),
                    np.full(len(models), b / budget),
                    np.full(len(models), len(available) / len(remaining)),
                    entropy,
                    np.full(len(models), anchor.gate.trust),
                    np.full(len(models), brier),
                    np.full(len(models), (budget - len(queries)) / budget),
                )
            )
            result.append(
                torch.cat(
                    (
                        points.mean(1),
                        pooled,
                        torch.as_tensor(extra, dtype=torch.float32, device=device),
                    ),
                    -1,
                )
            )
        return torch.stack(result, 1)

    def forward(self, features):
        raw = self.head(features)
        return torch.stack(
            (
                raw[..., 0],
                F.softplus(raw[..., 1]) + 1e-3,
                F.softplus(raw[..., 2]) + 1e-3,
            ),
            -1,
        )


def training_loss(parameters, lengths, truth):
    n = torch.as_tensor(lengths, device=parameters.device, dtype=parameters.dtype)
    y = torch.as_tensor(truth, device=parameters.device, dtype=parameters.dtype)
    logp = count_log_prob(parameters, n, y)
    mean = expected_count(parameters, n)
    mse = ((mean - y) / n).square().mean()
    monotonic = (
        ((mean[..., :-1] - mean[..., 1:]).clamp_min(0) / n[..., 1:]).square().mean()
        if len(lengths) > 1
        else mean.sum() * 0
    )
    return -logp.mean() + 0.2 * mse + 0.1 * monotonic
