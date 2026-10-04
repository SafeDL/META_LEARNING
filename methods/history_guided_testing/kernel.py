"""Small physical-coordinate kernels, without unused historical neural features."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import FEEDBACK


MODES = ("adaptive", "fixed_multiscale", "single_scale", "global_feedback")


def inverse_softplus(value):
    return math.log(math.expm1(value))


def matern52(distance, length):
    scaled = math.sqrt(5) * distance / length
    return (1 + scaled + scaled.square() / 3) * torch.exp(-scaled)


class RiskKernel(nn.Module):
    def __init__(self, mode="adaptive"):
        super().__init__()
        self.mode = mode
        self.raw_local_sigma = nn.Parameter(torch.tensor(inverse_softplus(.09)))
        self.raw_shared_sigma = nn.Parameter(torch.tensor(inverse_softplus(.25)))
        self.raw_local_length = nn.Parameter(torch.tensor(inverse_softplus(.20)))
        if mode != "single_scale":
            self.raw_lengths = nn.Parameter(torch.tensor([inverse_softplus(v) for v in (.05, .15, .5)]))
            if mode == "fixed_multiscale":
                self.raw_weights = nn.Parameter(torch.zeros(3))
            else:
                self.gate = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 3))
        self.double()

    def weights(self, x):
        raw = self.raw_weights.expand(len(x), -1) if self.mode == "fixed_multiscale" else self.gate(x)
        return F.normalize(F.softplus(raw), dim=-1)

    def covariance(self, inputs):
        parameter = next(self.parameters())
        x = torch.as_tensor(inputs, device=parameter.device, dtype=parameter.dtype)
        distance = torch.cdist(x, x)
        local_sigma = .03 + F.softplus(self.raw_local_sigma)
        shared_sigma = F.softplus(self.raw_shared_sigma)
        local = matern52(distance, F.softplus(self.raw_local_length))
        if self.mode == "single_scale":
            covariance = (local_sigma.square() + shared_sigma.square()) * local
        else:
            weights = self.weights(x)
            mixture = sum(weights[:, i:i + 1] * weights[:, i:i + 1].T * matern52(distance, length)
                          for i, length in enumerate(F.softplus(self.raw_lengths)))
            covariance = local_sigma.square() * local + shared_sigma.square() * mixture
        covariance = covariance * (x[:, 4:5] == x[:, 4:5].T)
        if self.mode != "global_feedback":
            taper = (1 - torch.abs(x[:, None, :4] - x[None, :, :4]) / FEEDBACK["radius"]).clamp(min=0).prod(-1)
            covariance = covariance * taper
        return covariance


def conditional_risk(kernel, x, prior, observed, count):
    covariance = kernel.covariance(x)
    prior = torch.as_tensor(prior, device=covariance.device, dtype=covariance.dtype)
    mean, query_covariance = prior[count:], covariance[count:, count:]
    if count:
        noise = (FEEDBACK["noise_variance"] + 1e-8) * torch.eye(count, device=covariance.device)
        factor = torch.linalg.cholesky(covariance[:count, :count] + noise)
        cross = covariance[count:, :count]
        residual = torch.as_tensor(observed, device=prior.device, dtype=prior.dtype) - prior[:count]
        mean = mean + (cross @ torch.cholesky_solve(residual[:, None], factor)).ravel()
        solved = torch.linalg.solve_triangular(factor, cross.T, upper=False)
        query_covariance = query_covariance - solved.T @ solved
    return mean, .5 * (query_covariance + query_covariance.T)
