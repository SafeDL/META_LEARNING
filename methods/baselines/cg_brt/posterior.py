"""Six-source Student risk posterior and a shared Gaussian reference."""
import numpy as np
from scipy.special import log_ndtr, logsumexp, stdtr
from scipy.stats import gamma

from methods.baselines.hgt.gp import RiskGP
from methods.baselines.saem.posterior import student_log_density

from .config import JITTER, NOISE, PRIOR_DF, SCALE_NODES, THRESHOLD
from .ranking import bivariate_normal_cdf


class RiskPosterior(RiskGP):
    def __init__(self, covariance, priors, probability, fixed_prior):
        super().__init__(covariance, fixed_prior, noise=NOISE)
        self.source_mean = np.asarray(priors, dtype=float).copy()
        self.log_weight = np.log(np.asarray(probability, dtype=float))
        self.log_weight -= logsumexp(self.log_weight)
        self.alpha = PRIOR_DF / 2
        self.beta = np.full(self.source_mean.shape[1], (PRIOR_DF - 2) / 2)
        nodes, weights = np.polynomial.legendre.leggauss(SCALE_NODES)
        self.scale_quantiles, self.scale_weights = .5 * (nodes + 1), .5 * weights
        self.evidence = []

    @property
    def weights(self):
        return np.exp(self.log_weight)

    def log_probabilities(self):
        variance = self.variance + NOISE + JITTER
        source_scale = np.sqrt(variance[:, None] * self.beta / self.alpha)
        source_z = (self.source_mean - THRESHOLD) / source_scale
        transfer = np.asarray([
            logsumexp(self.log_weight + np.log(stdtr(2 * self.alpha, sign * source_z)), axis=1)
            for sign in (1, -1)
        ])
        reference_z = (self.mean - THRESHOLD) / np.sqrt(variance)
        reference = np.asarray([log_ndtr(reference_z), log_ndtr(-reference_z)])
        return transfer, reference

    def observe(self, index, risk):
        residual = risk - self.source_mean[index]
        denominator = self.variance[index] + NOISE + JITTER
        log_density = student_log_density(
            residual, self.beta / self.alpha * denominator, 2 * self.alpha
        )
        before = self.weights.copy()
        super().observe(index, risk)
        root = self.factor[-1, -1]
        self.source_mean += self.solved_cross[:, -1, None] * residual / root
        self.log_weight += log_density
        self.log_weight -= logsumexp(self.log_weight)
        self.beta += .5 * residual ** 2 / denominator
        self.alpha += .5
        self.evidence.append({
            "index": int(index), "source_weights_before": before.tolist(),
            "source_weights_after": self.weights.tolist(),
            "risk_log_density": log_density.tolist(),
            "scale_mean_after": (self.beta / (self.alpha - 1)).tolist(),
        })

    def covariance_block(self, indices):
        indices = np.asarray(indices, dtype=int)
        solved = self.solved_cross[indices]
        block = self.covariance[np.ix_(indices, indices)] - solved @ solved.T
        return .5 * (block + block.T) + (NOISE + JITTER) * np.eye(len(indices))

    def draws(self, indices, count, generator, trust):
        indices = np.asarray(indices, dtype=int)
        factor = np.linalg.cholesky(self.covariance_block(indices))
        standard = generator.standard_normal((count, len(indices))) @ factor.T
        values = self.mean[indices] + standard
        transferred = np.flatnonzero(generator.random(count) < trust)
        if len(transferred):
            source = generator.choice(len(self.weights), len(transferred), p=self.weights)
            scale = self.beta[source] / generator.gamma(self.alpha, size=len(transferred))
            values[transferred] = (
                self.source_mean[indices][:, source].T
                + np.sqrt(scale[:, None]) * standard[transferred]
            )
        return values

    def discordance(self, first, second):
        indices = np.asarray([first, second], dtype=int)
        covariance = self.covariance_block(indices)
        deviation = np.sqrt(np.diag(covariance))
        correlation = covariance[0, 1] / np.prod(deviation)
        reference_z = (self.mean[indices] - THRESHOLD) / deviation
        reference = np.asarray([
            bivariate_normal_cdf(reference_z[0], -reference_z[1], -correlation),
            bivariate_normal_cdf(-reference_z[0], reference_z[1], -correlation),
        ])
        precision = gamma.ppf(self.scale_quantiles, a=self.alpha)
        scales = np.sqrt(self.beta[:, None] / precision)
        z = (self.source_mean[indices] - THRESHOLD) / deviation[:, None]
        first_z, second_z = z[0, :, None] / scales, z[1, :, None] / scales
        transfer = np.asarray([
            (bivariate_normal_cdf(first_z, -second_z, -correlation)
             @ self.scale_weights) @ self.weights,
            (bivariate_normal_cdf(-first_z, second_z, -correlation)
             @ self.scale_weights) @ self.weights,
        ])
        return transfer, reference
