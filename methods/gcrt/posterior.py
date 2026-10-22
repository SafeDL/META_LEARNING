"""Historical-source and independent-target Bernoulli GP components."""

import numpy as np
from scipy.special import logsumexp

from methods.common.classification import logistic_moments, site_derivatives


class CollisionPosterior:
    def __init__(self, means, covariances, weights, *, fixed_weights=False,
                 component_names=None):
        self.mean = np.asarray(means, dtype=np.float64).copy()
        self.covariance = np.asarray(covariances, dtype=np.float64).copy()
        weights = np.asarray(weights, dtype=np.float64)
        if self.mean.ndim != 2 or self.covariance.shape != (
            self.mean.shape[0], self.mean.shape[1], self.mean.shape[1]
        ):
            raise ValueError("Component latent means and covariances do not align")
        if len(weights) != len(self.mean) or np.any(weights <= 0):
            raise ValueError("Each component requires positive prior weight")
        if not (np.isfinite(self.mean).all() and np.isfinite(self.covariance).all()):
            raise ValueError("Nonfinite latent prior")
        self.log_weight = np.log(weights / weights.sum())
        self.fixed_weights = fixed_weights
        self.observed = set()
        self.sites = []
        self.component_names = (list(component_names) if component_names is not None
                                else [f"component_{j}" for j in range(len(weights))])

    @property
    def weights(self):
        return np.exp(self.log_weight)

    def component_probabilities(self, indices=None):
        indices = np.arange(self.mean.shape[1]) if indices is None else np.asarray(indices)
        return logistic_moments(self.mean[:, indices],
                                self.covariance.diagonal(axis1=1, axis2=2)[:, indices])[0]

    def probabilities(self, indices=None):
        return self.weights @ self.component_probabilities(indices)

    def conditional_probabilities(self, index, label, indices):
        """Virtual ADF update; this method never changes evidence or arrays."""
        _, gradient, curvature = site_derivatives(
            self.mean[:, index], self.covariance[:, index, index], label)
        indices = np.asarray(indices, dtype=int)
        cross = self.covariance[:, indices, index]
        mean = self.mean[:, indices] + gradient[:, None] * cross
        variance = (self.covariance.diagonal(axis1=1, axis2=2)[:, indices]
                    + curvature[:, None] * cross**2)
        p = logistic_moments(mean, variance)[0]
        return self._conditional_weights(index, label) @ p

    def _conditional_weights(self, index, label):
        if self.fixed_weights:
            return self.weights
        p = self.component_probabilities([index])[:, 0]
        log_weight = self.log_weight + np.log(np.maximum(p if label else 1 - p, 1e-15))
        return np.exp(log_weight - logsumexp(log_weight))

    def observe(self, index, collision):
        if index in self.observed or not 0 <= index < self.mean.shape[1]:
            raise ValueError("Invalid or repeated target query index")
        log_likelihood, gradient, curvature = site_derivatives(
            self.mean[:, index], self.covariance[:, index, index], collision)
        # Evidence uses the predictive likelihood before the site update.
        weights = self._conditional_weights(index, collision)
        cross = self.covariance[:, :, index].copy()
        self.sites.append({"index": int(index), "gradient": gradient.copy(),
                           "curvature": curvature.copy(), "cross": cross.copy()})
        self.mean += gradient[:, None] * cross
        self.covariance += curvature[:, None, None] * cross[:, :, None] * cross[:, None, :]
        if self.covariance.diagonal(axis1=1, axis2=2).min() < -1e-9:
            raise FloatingPointError("Negative Bernoulli GP variance")
        evidence = float(logsumexp(self.log_weight + log_likelihood))
        self.log_weight = np.log(np.maximum(weights, 1e-300))
        self.log_weight -= logsumexp(self.log_weight)
        self.observed.add(int(index))
        return evidence

    def extend(self, means, cross_covariances, covariances):
        """Condition new prior coordinates on the same paid sites, without replay calls."""
        means = np.asarray(means, dtype=np.float64).copy()
        cross = np.asarray(cross_covariances, dtype=np.float64).copy()
        covariance = np.asarray(covariances, dtype=np.float64).copy()
        components, old_size = self.mean.shape
        if (means.ndim != 2 or means.shape[0] != components or not means.shape[1]
                or cross.shape != (components, means.shape[1], old_size)
                or covariance.shape != (components, means.shape[1], means.shape[1])):
            raise ValueError("New-point prior blocks do not match the current component geometry")
        for site in self.sites:
            to_query = cross[:, :, site["index"]].copy()
            gradient, curvature = site["gradient"], site["curvature"]
            means += gradient[:, None] * to_query
            covariance += curvature[:, None, None] * to_query[:, :, None] * to_query[:, None, :]
            cross += curvature[:, None, None] * to_query[:, :, None] * site["cross"][:, None, :]
            site["cross"] = np.concatenate((site["cross"], to_query), axis=1)
        combined = np.concatenate((
            np.concatenate((self.covariance, cross.transpose(0, 2, 1)), axis=2),
            np.concatenate((cross, covariance), axis=2)), axis=1)
        if combined.diagonal(axis1=1, axis2=2).min() < -1e-9:
            raise FloatingPointError("Negative variance after posterior expansion")
        self.mean = np.concatenate((self.mean, means), axis=1)
        self.covariance = combined

    def retain(self, indices):
        """Discard only unqueried proposals, retaining every paid conditioning site."""
        indices = list(map(int, indices))
        if (len(set(indices)) != len(indices) or not self.observed.issubset(indices)
                or any(not 0 <= i < self.mean.shape[1] for i in indices)):
            raise ValueError("Marginalization must retain every paid observation exactly once")
        mapping = {old: new for new, old in enumerate(indices)}
        self.mean = self.mean[:, indices].copy()
        self.covariance = self.covariance[:, indices][:, :, indices].copy()
        self.observed = {mapping[index] for index in self.observed}
        for site in self.sites:
            site["index"] = mapping[site["index"]]
            site["cross"] = site["cross"][:, indices].copy()
