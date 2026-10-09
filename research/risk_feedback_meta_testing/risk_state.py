"""Gaussian moment filtering for a risk score censored to [0, 1]."""
import math

import numpy as np
import torch

from methods.history_guided_testing.kernel import matern52


def risk_kernel(x, discrepancy):
    family = x[:, 4].long()
    variance = x.new_tensor(discrepancy["gp_variance"])[family]
    length = x.new_tensor(discrepancy["length"])[family]
    covariance = (variance.sqrt()[:, None] * variance.sqrt()[None, :]
                  * matern52(torch.cdist(x[:, :4], x[:, :4]), length[:, None])
                  * (family[:, None] == family[None, :]))
    noise = x.new_tensor(discrepancy["noise_variance"])[family]
    return covariance, noise


def observation_moments(mean, variance, observation):
    """Return log likelihood, innovation and covariance reduction fraction."""
    standard_deviation = variance.sqrt()
    if observation == 1:
        threshold = (1 - mean) / standard_deviation
        log_likelihood = torch.special.log_ndtr(-threshold)
        ratio = torch.exp(-0.5 * threshold.square() - 0.5 * math.log(2 * math.pi)
                          - log_likelihood)
        innovation = standard_deviation * ratio
        reduction = ratio * (ratio - threshold)
    elif observation == 0:
        threshold = -mean / standard_deviation
        log_likelihood = torch.special.log_ndtr(threshold)
        ratio = torch.exp(-0.5 * threshold.square() - 0.5 * math.log(2 * math.pi)
                          - log_likelihood)
        innovation = -standard_deviation * ratio
        reduction = ratio * (ratio + threshold)
    else:
        innovation = observation - mean
        log_likelihood = -0.5 * (innovation.square() / variance
                                 + torch.log(2 * math.pi * variance))
        reduction = torch.ones_like(mean)
    return log_likelihood, innovation, reduction.clamp(0, 1)


class CensoredRiskState:
    """Track behavior-specific covariances with low-rank update factors.

    Interior observations use exact Gaussian conditioning. Boundary updates
    retain the first two moments of one clipped-normal observation and then
    approximate the latent field as Gaussian for later filtering steps.
    """

    def __init__(self, x, predictions, discrepancy, observation_capacity):
        self.baseline = predictions.clone()
        self.mean = predictions.clone()
        self.covariance, self.noise = risk_kernel(x, discrepancy)
        self.variance = self.covariance.diagonal()[None, :].expand_as(
            predictions).clone()
        self.log_weights = predictions.new_full(
            (len(predictions),), -math.log(len(predictions)))
        self.factors = predictions.new_empty(
            (observation_capacity, *predictions.shape))
        self.count = 0

    def cross_covariance(self, index):
        cross = self.covariance[:, index][None, :].expand_as(self.mean).clone()
        if self.count:
            factors = self.factors[:self.count]
            cross -= torch.einsum("tmn,tm->mn", factors, factors[:, :, index])
        return cross

    def conditional_moments(self, index, risk):
        variance = self.variance[:, index] + self.noise[index]
        likelihood, innovation, reduction = observation_moments(
            self.mean[:, index], variance, risk)
        weights = self.log_weights + likelihood
        weights -= torch.logsumexp(weights, 0)
        cross = self.cross_covariance(index)
        mean = self.mean + cross * (innovation / variance)[:, None]
        factor = cross * (reduction / variance).sqrt()[:, None]
        updated_variance = (self.variance - factor.square()).clamp_min(0)
        return weights, mean, updated_variance, factor

    def observe(self, index, risk):
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Risk observation must be finite and in [0, 1]")
        weights, mean, variance, factor = self.conditional_moments(index, risk)
        self.factors[self.count] = factor
        self.log_weights, self.mean, self.variance = weights, mean, variance
        self.count += 1

    def conditional_candidates(self, index, nodes):
        """Compute hypothetical moments in a batch without storing observations."""
        mean = self.mean[:, index]
        variance = self.variance[:, index] + self.noise[index]
        innovation = nodes[:, None] - mean[None, :]
        likelihood = -0.5 * (innovation.square() / variance[None, :]
                             + torch.log(2 * math.pi * variance)[None, :])
        reduction = torch.ones_like(likelihood)
        for endpoint in (0, 1):
            log_mass, shift, fraction = observation_moments(mean, variance, endpoint)
            mask = (nodes == endpoint)[:, None]
            likelihood = torch.where(mask, log_mass[None, :], likelihood)
            innovation = torch.where(mask, shift[None, :], innovation)
            reduction = torch.where(mask, fraction[None, :], reduction)
        weights = self.log_weights[None, :] + likelihood
        weights -= torch.logsumexp(weights, -1, keepdim=True)
        cross = self.cross_covariance(index)
        updated_mean = (self.mean[None, :, :]
                        + cross[None, :, :] * (innovation / variance)[..., None])
        updated_variance = (self.variance[None, :, :]
                            - cross.square()[None, :, :]
                            * (reduction / variance)[..., None]).clamp_min(0)
        return weights, updated_mean, updated_variance

    def predictive_nodes(self, index, interior_count):
        """Integrate endpoint atoms exactly and the interior in CDF space."""
        mean = self.mean[:, index]
        scale = (self.variance[:, index] + self.noise[index]).sqrt()
        weights = self.log_weights.exp()
        lower = (weights * torch.special.ndtr(-mean / scale)).sum()
        upper = (weights * torch.special.ndtr((mean - 1) / scale)).sum()
        interior = (1 - lower - upper).clamp_min(0)
        points, quadrature_weights = np.polynomial.legendre.leggauss(interior_count)
        points = mean.new_tensor(points)
        quadrature_weights = mean.new_tensor(quadrature_weights)
        probabilities = lower + interior * (points + 1) / 2
        low, high = torch.zeros_like(points), torch.ones_like(points)
        for _ in range(32):
            midpoint = (low + high) / 2
            cdf = (weights[:, None] * torch.special.ndtr(
                (midpoint[None, :] - mean[:, None]) / scale[:, None])).sum(0)
            below = cdf < probabilities
            low = torch.where(below, midpoint, low)
            high = torch.where(below, high, midpoint)
        nodes = torch.cat((mean.new_tensor([0.]), (low + high) / 2,
                           mean.new_tensor([1.])))
        probability_weights = torch.cat((lower[None],
                                         interior * quadrature_weights / 2,
                                         upper[None]))
        return nodes, probability_weights
