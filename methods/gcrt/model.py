"""Independent invariant set encoder and remaining-discovery count model."""

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit, logit, logsumexp
from torch import nn
from torch.nn import functional as F

from methods.common.counts import count_cdf, count_log_prob, expected_count


def snapshot(model, remaining, queries, evidence, budget, initial_indices, index):
    available = np.flatnonzero(remaining)
    components = model.component_probabilities(available)
    recent = [
        (float(q["risk"] > 0.5) - e["prequery_probability"]) ** 2
        for q, e in zip(queries[-10:], evidence[-10:])
    ]
    return {
        "x": model.prior.x[available].astype(np.float32),
        "tokens": model.source_tokens(available),
        "q": (components @ model.weights).astype(np.float32),
        "target_q": components[:, -1].astype(np.float32),
        "available": available,
        "next_index": int(index),
        "pending_initial": np.asarray(initial_indices[len(queries) :], dtype=int),
        "step": len(queries),
        "found": sum(q["risk"] > 0.5 for q in queries),
        "budget": budget,
        "pool_size": len(remaining),
        "brier": float(np.mean(recent)) if recent else 0.0,
    }


class RemainingDiscoveries(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()
        )
        self.point = nn.Sequential(
            nn.Linear(39, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(140, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def features(self, state, lengths):
        device = next(self.parameters()).device
        if "encoded" in state:
            positions = [state["encoded_lengths"].index(b) for b in lengths]
            return state["encoded"][positions].to(device)
        tokens = torch.as_tensor(state["tokens"], device=device, dtype=torch.float32)
        embedded = (self.source(tokens) * tokens[..., 2, None]).sum(-2)
        x = torch.as_tensor(state["x"], device=device, dtype=torch.float32)
        q = torch.as_tensor(state["q"], device=device, dtype=torch.float32)
        target = torch.as_tensor(state["target_q"], device=device, dtype=torch.float32)
        points = self.point(torch.cat((embedded, x, q[:, None], target[:, None]), -1))
        weights = state["tokens"][0, :, 2]
        entropy = float(-(weights * np.log(np.maximum(weights, 1e-30))).sum())
        entropy /= np.log(len(weights))
        available = state["available"]
        position = {int(i): j for j, i in enumerate(available)}
        ordered = np.argsort(-state["q"], kind="stable")
        initial = [position[int(i)] for i in state["pending_initial"]]
        rows = []
        for b in lengths:
            if b == 1:
                top = [position[state["next_index"]]]
            else:
                fixed = initial[:b]
                fixed_set = set(fixed)
                top = (
                    fixed
                    + [int(i) for i in ordered if int(i) not in fixed_set][
                        : b - len(fixed)
                    ]
                )
            extra = points.new_tensor(
                [
                    state["found"] / state["budget"],
                    state["step"] / state["budget"],
                    b / state["budget"],
                    len(available) / state["pool_size"],
                    entropy,
                    float(weights[state["tokens"][0, :, 3] > 0].sum()),
                    state["brier"],
                    len(initial) / state["budget"],
                    float(state["q"][top].mean()),
                    np.log1p(state["budget"]) / np.log(201),
                    np.log1p(state["pool_size"]) / np.log(1025),
                    np.log1p(b) / np.log(201),
                ]
            )
            rows.append(torch.cat((points.mean(0), points[top].mean(0), extra)))
        return torch.stack(rows)

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

    def predict(self, state, lengths):
        return self(self.features(state, lengths))

    def freeze_representation(self):
        self.source.requires_grad_(False)
        self.point.requires_grad_(False)


def loss_terms(parameters, lengths, truth, found=0):
    n = parameters.new_tensor(lengths)
    y = parameters.new_tensor(truth)
    mean = expected_count(parameters, n)
    supervised = -count_log_prob(parameters, n, y)
    probability = count_cdf(parameters[-1], lengths[-1], found // 19)
    event = (y[-1] <= found // 19).to(parameters.dtype)
    auxiliary = torch.zeros_like(supervised)
    auxiliary[-1] = 0.2 * len(lengths) * (probability - event).square()
    supervised = supervised + auxiliary
    monotonic = ((mean[:-1] - mean[1:]).clamp_min(0) / n[1:]).square()
    return supervised, 0.1 * monotonic


def weighted_loss(
    parameters, lengths, truth, visible=None, probability=None, *, found=0
):
    supervised, monotonic = loss_terms(parameters, lengths, truth, found)
    if visible is None:
        return supervised.mean() + (monotonic.mean() if len(monotonic) else 0)
    weight = parameters.new_tensor(visible) / parameters.new_tensor(probability)
    result = (supervised * weight).mean()
    if len(monotonic):
        result += (monotonic * weight[1:]).mean()
    return result


def log_pmf(parameters, length):
    k = torch.arange(length + 1, dtype=torch.float64, device=parameters.device)
    p = count_log_prob(parameters.double()[None], length, k)
    return p - p.logsumexp(0)


def tilted_log_pmf(base, u):
    score = base + u * torch.arange(len(base), dtype=base.dtype, device=base.device)
    return score - score.logsumexp(0)


def corrected_log_pmf(base, u):
    """Minimum-KL mean projection with a budget-independent rate-logit shift."""
    if u == 0 or len(base) == 1:
        return base
    values = base.detach().cpu().numpy().astype(float)
    k = np.arange(len(values), dtype=float)
    n = len(k) - 1
    rate = float(np.exp(values) @ k / n)
    target = n * expit(logit(np.clip(rate, 1e-12, 1 - 1e-12)) + u)

    def residual(coefficient):
        scores = values + coefficient * k
        return float(np.exp(scores - logsumexp(scores)) @ k - target)

    left, right = -1.0, 1.0
    while residual(left) > 0:
        left *= 2
    while residual(right) < 0:
        right *= 2
    coefficient = brentq(residual, left, right, xtol=1e-12)
    return tilted_log_pmf(base, coefficient)
