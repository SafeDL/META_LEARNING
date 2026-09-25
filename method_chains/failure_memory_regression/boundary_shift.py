"""One-dimensional posterior for a local cross-version boundary shift."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from method_chains.failure_memory_regression.boundary_memory import Patch


class BoundaryShift:
    def __init__(self, patches: list[Patch]) -> None:
        self.grid = np.linspace(-0.30, 0.30, 81)
        prior = -0.5 * (self.grid / 0.10) ** 2
        prior -= logsumexp(prior)
        self.log_posterior = {patch.patch_id: prior.copy() for patch in patches}

    def probabilities(self, patch: Patch, u: float) -> np.ndarray:
        tau = max(patch.width / 2, 0.02)
        return 0.02 + 0.96 * expit((self.grid - u) / tau)

    def score(self, patch: Patch, u: float, v: float) -> float:
        posterior = np.exp(self.log_posterior[patch.patch_id])
        probability = float(posterior @ self.probabilities(patch, u))
        width = max(2 * patch.width, 0.10)
        return probability * float(np.exp(-v * v / (2 * width * width)))

    def update(self, patch: Patch, u: float, collision: bool) -> None:
        probabilities = self.probabilities(patch, u)
        log_values = self.log_posterior[patch.patch_id]
        log_values += np.log(probabilities) if collision else np.log1p(-probabilities)
        log_values -= logsumexp(log_values)
