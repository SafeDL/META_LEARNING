"""Neural response likelihood and a shared latent-field failure model."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import LOGIT_TO_PROBIT_SCALE, MIN_STANDARD_DEVIATION


def risk_log_likelihood(mean, variance, risk):
    """Normal density inside the interval, tail probability at its endpoints."""
    scale = variance.sqrt()
    interior = -0.5 * ((risk - mean).square() / variance
                      + torch.log(2 * math.pi * variance))
    result = torch.where(risk == 1, torch.special.log_ndtr((mean - 1) / scale),
                         interior)
    return torch.where(risk == 0, torch.special.log_ndtr(-mean / scale), result)


def failure_log_probabilities(mean, family, log_weights, field_mean,
                              field_variance, sensitivity):
    mean, log_weights, field_mean, field_variance, sensitivity = (
        value.double() for value in
        (mean, log_weights, field_mean, field_variance, sensitivity)
    )
    slope = sensitivity[family]
    standardized = (mean + slope * field_mean) / torch.sqrt(
        1 + slope.square() * field_variance)
    collision = torch.logsumexp(
        log_weights[:, None] + torch.special.log_ndtr(standardized), 0)
    safe = torch.logsumexp(
        log_weights[:, None] + torch.special.log_ndtr(-standardized), 0)
    return collision.clamp_max(0), safe.clamp_max(0)


class ResponsePrior(nn.Module):
    """The same architecture and initialization in both learning modes."""

    def __init__(self, weight, bias, center, scale, discrepancy):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center, dtype=weight.dtype,
                                                      device=weight.device))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=weight.dtype,
                                                     device=weight.device))
        self.risk_corrections = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 32),
                          nn.SiLU(), nn.Linear(32, 1)) for _ in (0, 1)
        ])
        self.amplitude_heads = nn.ModuleList([nn.Linear(128, 1) for _ in (0, 1)])
        self.noise_heads = nn.ModuleList([nn.Linear(128, 1) for _ in (0, 1)])
        self.failure_weight = nn.Parameter(
            weight.detach().clone() * self.scale / LOGIT_TO_PROBIT_SCALE)
        self.failure_bias = nn.Parameter(
            (bias.detach().clone() + (weight * self.center).sum(-1))
            / LOGIT_TO_PROBIT_SCALE)
        self.sensitivity = nn.Parameter(bias.new_zeros(2))
        for family in (0, 1):
            nn.init.zeros_(self.risk_corrections[family][-1].weight)
            nn.init.zeros_(self.risk_corrections[family][-1].bias)
            for head, key in ((self.amplitude_heads[family], "gp_variance"),
                              (self.noise_heads[family], "noise_variance")):
                nn.init.zeros_(head.weight)
                standard_deviation = math.sqrt(discrepancy[key][family])
                value = max(standard_deviation - MIN_STANDARD_DEVIATION,
                            MIN_STANDARD_DEVIATION)
                nn.init.constant_(head.bias, math.log(math.expm1(value)))

    def normalized_features(self, features, family):
        return (features - self.center[family]) / self.scale[family]

    def risk_fields(self, features, family, frozen_risk):
        normalized = self.normalized_features(features, family)
        means = torch.empty_like(frozen_risk, dtype=torch.float64)
        amplitudes, noises = torch.empty_like(means), torch.empty_like(means)
        for value in (0, 1):
            selected = family == value
            hidden = normalized[:, selected]
            means[:, selected] = (frozen_risk[:, selected].double()
                                 + self.risk_corrections[value](hidden)[..., 0].double())
            amplitudes[:, selected] = (
                F.softplus(self.amplitude_heads[value](hidden)[..., 0]).double()
                + MIN_STANDARD_DEVIATION)
            noises[:, selected] = (
                F.softplus(self.noise_heads[value](hidden)[..., 0]).double()
                + MIN_STANDARD_DEVIATION)
        return means, amplitudes, noises

    def failure_means(self, features, family):
        hidden = self.normalized_features(features, family)
        return ((hidden * self.failure_weight[family]).sum(-1)
                + self.failure_bias[family]).double()

    def failure_logs(self, features, family, log_weights, field_mean,
                     field_variance):
        return failure_log_probabilities(
            self.failure_means(features, family), family, log_weights,
            field_mean, field_variance, self.sensitivity)

