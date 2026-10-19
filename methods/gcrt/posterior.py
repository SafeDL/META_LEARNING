"""Student-t components without a Gaussian reference or ranking state."""

import numpy as np
from scipy.special import logsumexp, stdtr

from methods.baselines.saem.posterior import student_log_density
from methods.common.config import JITTER, NOISE


class StudentComponent:
    def __init__(self, covariance, means):
        self.covariance = np.asarray(covariance)
        self.mean = np.asarray(means, dtype=float).copy()
        self.variance = np.diag(self.covariance).copy()
        self.solved = np.empty((len(self.mean), 0))
        self.alpha = 2.5
        self.beta = np.full(self.mean.shape[1], 1.5)

    def conditional(self, index, values, indices):
        values = np.asarray(values).reshape(-1)
        denominator = self.variance[index] + NOISE + JITTER
        cross = (
            self.covariance[indices, index] - self.solved[indices] @ self.solved[index]
        )
        residual = values[:, None] - self.mean[index]
        mean = self.mean[indices][None] + (
            cross[None, :, None] * residual[:, None] / denominator
        )
        variance = self.variance[indices] + NOISE + JITTER - cross**2 / denominator
        beta = self.beta + 0.5 * residual**2 / denominator
        density = student_log_density(
            residual, self.beta / self.alpha * denominator, 2 * self.alpha
        )
        return mean, variance, self.alpha + 0.5, beta, density

    def observe(self, index, value):
        denominator = self.variance[index] + NOISE + JITTER
        if denominator <= 0:
            raise FloatingPointError("Nonpositive Student conditional variance")
        residual = value - self.mean[index]
        density = student_log_density(
            residual, self.beta / self.alpha * denominator, 2 * self.alpha
        )
        cross = self.covariance[:, index] - self.solved @ self.solved[index]
        self.mean += cross[:, None] * residual / denominator
        self.variance -= cross**2 / denominator
        if self.variance.min() < -1e-8:
            raise FloatingPointError("Negative conditional variance")
        self.variance = np.maximum(self.variance, 0)
        self.solved = np.column_stack((self.solved, cross / np.sqrt(denominator)))
        self.beta += 0.5 * residual**2 / denominator
        self.alpha += 0.5
        return density


class JointPosterior:
    def __init__(self, prior, kappa=1.0, target_probability=0.25):
        if len(np.unique(prior.x[:, 4])) != 1:
            raise ValueError("Build each functional scenario independently")
        self.prior = prior
        self.groups = []
        self.count = len(prior.names) + 1
        self.log_weight = np.log(
            np.r_[prior.weights * (1 - target_probability), target_probability]
        )
        geometry = {}
        for j, archive in enumerate(prior.archive_covariances):
            geometry.setdefault(id(archive), ([], archive))[0].append(j)
        for columns, archive in geometry.values():
            self.groups.append(
                (
                    np.asarray(columns),
                    StudentComponent(
                        prior.enhanced_covariance + kappa * archive,
                        prior.means[:, columns],
                    ),
                )
            )
        self.groups.append(
            (
                np.array([self.count - 1]),
                StudentComponent(
                    prior.physical_covariance, np.full((len(prior.x), 1), 0.5)
                ),
            )
        )
        self.observed = set()

    @property
    def weights(self):
        return np.exp(self.log_weight)

    def parameters(self, indices):
        mean = np.empty((len(indices), self.count))
        variance = np.empty_like(mean)
        beta = np.empty(self.count)
        for columns, component in self.groups:
            mean[:, columns] = component.mean[indices]
            variance[:, columns] = component.variance[indices, None] + NOISE + JITTER
            beta[columns] = component.beta
        return mean, variance, self.groups[0][1].alpha, beta

    def component_probabilities(self, indices=None):
        indices = (
            np.arange(len(self.prior.x)) if indices is None else np.asarray(indices)
        )
        mean, variance, alpha, beta = self.parameters(indices)
        return stdtr(2 * alpha, (mean - 0.5) / np.sqrt(variance * beta / alpha))

    def probabilities(self):
        return self.component_probabilities() @ self.weights

    def source_tokens(self, indices):
        mean, variance, alpha, beta = self.parameters(indices)
        tokens = np.zeros((len(indices), self.count, 4), dtype=np.float32)
        tokens[..., 0] = mean
        tokens[..., 1] = np.sqrt(variance * beta / (alpha - 1))
        tokens[..., 2] = self.weights
        tokens[:, -1, 3] = 1
        return tokens

    def observe(self, index, risk):
        if index in self.observed or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid or repeated paid risk observation")
        density = np.empty(self.count)
        for columns, component in self.groups:
            density[columns] = component.observe(index, risk)
        self.log_weight += density
        evidence = float(logsumexp(self.log_weight))
        self.log_weight -= evidence
        self.observed.add(int(index))
        return evidence

    def conditional_probabilities(self, index, values, indices):
        mean = np.empty((len(values), len(indices), self.count))
        variance = np.empty((len(indices), self.count))
        beta = np.empty((len(values), self.count))
        density = np.empty_like(beta)
        for columns, component in self.groups:
            mu, var, alpha, scale, logp = component.conditional(index, values, indices)
            mean[:, :, columns] = mu
            variance[:, columns] = var[:, None]
            beta[:, columns] = scale
            density[:, columns] = logp
        weights = self.log_weight + density
        weights -= logsumexp(weights, axis=-1, keepdims=True)
        z = (mean - 0.5) / np.sqrt(variance[None] * beta[:, None] / alpha)
        return (stdtr(2 * alpha, z) * np.exp(weights[:, None])).sum(-1)

    def marginal_draws(self, indices, count, generator):
        # Common component, precision and normal draws couple candidate integrals.
        source = generator.choice(self.count, size=count, p=self.weights)
        mean, variance, alpha, beta = self.parameters(indices)
        precision = generator.gamma(alpha, size=count)
        normal = generator.standard_normal(count)
        return mean[:, source].T + normal[:, None] * np.sqrt(
            variance[:, source].T * beta[source, None] / precision[:, None]
        )
