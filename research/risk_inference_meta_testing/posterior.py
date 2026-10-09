"""Differentiable conditioning of a shared unit Gaussian field."""
import math

import numpy as np
import torch

from methods.history_guided_testing.kernel import matern52
from research.risk_feedback_meta_testing.risk_state import observation_moments

from .config import KERNEL_NUGGET


def correlation(x, lengths):
    family = x[:, 4].long()
    length = x.new_tensor(lengths)[family]
    matrix = (matern52(torch.cdist(x[:, :4], x[:, :4]), length[:, None])
              * (family[:, None] == family[None, :]))
    return matrix + KERNEL_NUGGET * torch.eye(len(x), dtype=x.dtype, device=x.device)


class UnitFieldState:
    """Functional low-rank updates preserve gradients through risk evidence."""

    def __init__(self, covariance, risk_mean, amplitude, noise):
        self.covariance = covariance
        self.risk_mean, self.amplitude, self.noise = risk_mean, amplitude, noise
        self.mean = torch.zeros_like(risk_mean)
        self.variance = covariance.diagonal()[None, :].expand_as(risk_mean)
        self.log_weights = risk_mean.new_full(
            (len(risk_mean),), -math.log(len(risk_mean)))
        self.factors = []

    def cross_covariance(self, index):
        cross = self.covariance[:, index][None, :].expand_as(self.mean)
        if self.factors:
            factors = torch.stack(self.factors)
            cross = cross - torch.einsum("tmn,tm->mn", factors, factors[:, :, index])
        return cross

    def observe(self, index, risk):
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("A risk observation must be finite and in [0, 1]")
        amplitude = self.amplitude[:, index]
        mean = self.risk_mean[:, index] + amplitude * self.mean[:, index]
        variance = amplitude.square() * self.variance[:, index] + self.noise[:, index].square()
        likelihood, innovation, reduction = observation_moments(mean, variance, risk)
        weights = self.log_weights + likelihood
        self.log_weights = weights - torch.logsumexp(weights, 0)
        cross = self.cross_covariance(index) * amplitude[:, None]
        self.mean = self.mean + cross * (innovation / variance)[:, None]
        factor = cross * (reduction / variance).sqrt()[:, None]
        self.variance = (self.variance - factor.square()).clamp_min(0)
        self.factors.append(factor)


def gaussian_condition(covariance, means, amplitudes, noises, support, risk):
    support = torch.as_tensor(support, dtype=torch.long, device=means.device)
    values = torch.as_tensor(risk, dtype=means.dtype, device=means.device)
    amplitude = amplitudes[:, support]
    observed_covariance = (
        covariance[support][:, support][None, :, :]
        * amplitude[:, :, None] * amplitude[:, None, :]
        + torch.diag_embed(noises[:, support].square()))
    factor = torch.linalg.cholesky(observed_covariance)
    residual = values[None, :] - means[:, support]
    solved = torch.cholesky_solve(residual[..., None], factor)[..., 0]
    evidence = -0.5 * ((residual * solved).sum(-1)
                      + 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                      + len(support) * math.log(2 * math.pi))
    log_weights = evidence - torch.logsumexp(evidence, 0)
    cross = covariance[:, support][None, :, :] * amplitude[:, None, :]
    mean = (cross * solved[:, None, :]).sum(-1)
    reduction = (cross * torch.cholesky_solve(cross.transpose(-2, -1), factor)
                 .transpose(-2, -1)).sum(-1)
    variance = (covariance.diagonal()[None, :] - reduction).clamp_min(0)
    return log_weights, mean, variance


def condition_field(x, risk_mean, amplitude, noise, support, feedback, lengths):
    """No query outcomes or true target parameters enter this function."""
    covariance = correlation(x, lengths)
    if not len(support):
        weights = risk_mean.new_full((len(risk_mean),), -math.log(len(risk_mean)))
        return (weights, torch.zeros_like(risk_mean),
                covariance.diagonal()[None, :].expand_as(risk_mean))
    if np.all((np.asarray(feedback) > 0) & (np.asarray(feedback) < 1)):
        return gaussian_condition(covariance, risk_mean, amplitude, noise,
                                  support, feedback)
    selected = torch.as_tensor(support, dtype=torch.long, device=x.device)
    state = UnitFieldState(covariance[selected][:, selected], risk_mean[:, selected],
                           amplitude[:, selected], noise[:, selected])
    for number, risk in enumerate(feedback):
        state.observe(number, float(risk))
    # G(query) conditional on G(support) is unchanged by observations of R(support).
    factor = torch.linalg.cholesky(covariance[selected][:, selected])
    projection = torch.cholesky_solve(covariance[:, selected].T, factor).T
    mean = state.mean @ projection.T
    projected_factors = torch.stack(state.factors) @ projection.T
    variance = (covariance.diagonal()[None, :]
                - projected_factors.square().sum(0)).clamp_min(0)
    return state.log_weights, mean, variance


def point_condition(mean, amplitude, noise, observed, prior_variance):
    """Historical ordinary supervision conditions on one queried risk per point."""
    variance = amplitude.square() * prior_variance + noise.square()
    innovation = observed - mean
    reduction = torch.ones_like(mean)
    for endpoint in (0, 1):
        _, shift, fraction = observation_moments(mean, variance, endpoint)
        mask = observed == endpoint
        innovation = torch.where(mask, shift, innovation)
        reduction = torch.where(mask, fraction, reduction)
    cross = amplitude * prior_variance
    posterior_mean = cross * innovation / variance
    posterior_variance = (prior_variance - cross.square() * reduction / variance).clamp_min(0)
    return posterior_mean, posterior_variance
