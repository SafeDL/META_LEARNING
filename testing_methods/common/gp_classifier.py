"""Differentiable Laplace Bernoulli GP for target-only and deep-kernel baselines."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .classification import torch_moments


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


def laplace(model, x, labels, *, iterations=10):
    x = torch.as_tensor(x, dtype=torch.float64, device=model.mean.device)
    labels = torch.as_tensor(labels, dtype=torch.float64, device=x.device)
    eye = torch.eye(len(x), dtype=torch.float64, device=x.device)
    covariance = model.covariance(x) + 1e-5 * eye
    f = model.mean.expand(len(x))
    for _ in range(iterations):
        p = f.sigmoid()
        weight = (p * (1 - p)).clamp_min(1e-9)
        root = weight.sqrt()
        factor = torch.linalg.cholesky(eye + root[:, None] * covariance * root[None, :])
        b = weight * (f - model.mean) + labels - p
        solved = torch.cholesky_solve((root * (covariance @ b))[:, None], factor)[:, 0]
        alpha = b - root * solved
        f = model.mean + covariance @ alpha
    p = f.sigmoid()
    root = (p * (1 - p)).clamp_min(1e-9).sqrt()
    factor = torch.linalg.cholesky(eye + root[:, None] * covariance * root[None, :])
    prior_factor = torch.linalg.cholesky(covariance)
    alpha = torch.cholesky_solve((f - model.mean)[:, None], prior_factor)[:, 0]
    nll = (
        0.5 * (alpha * (f - model.mean)).sum()
        + F.binary_cross_entropy_with_logits(f, labels, reduction="sum")
        + factor.diagonal().log().sum()
    )
    return nll, alpha, root, factor


def collision_probabilities(model, support_x, labels, query_x):
    if not len(labels):
        x = torch.as_tensor(query_x, dtype=torch.float64, device=model.mean.device)
        return torch_moments(model.mean.expand(len(x)), model.covariance(x).diagonal())[0]
    _, alpha, root, factor = laplace(model, support_x, labels)
    cross = model.covariance(support_x, query_x)
    mean = model.mean + cross.T @ alpha
    solved = torch.linalg.solve_triangular(factor, root[:, None] * cross, upper=False)
    variance = (model.covariance(query_x).diagonal() - solved.square().sum(0)).clamp_min(1e-10)
    return torch_moments(mean, variance)[0]


def fit_target(model, x, labels, *, steps=10, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = laplace(model, x, labels)[0] / len(labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Bernoulli baseline loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
