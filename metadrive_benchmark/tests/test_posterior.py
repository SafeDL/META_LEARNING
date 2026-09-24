from __future__ import annotations

import numpy as np

from metadrive_benchmark.mining.posterior import LatentVulnerabilityPosterior


def test_posterior_update_reduces_uncertainty_in_observed_basis_direction() -> None:
    posterior = LatentVulnerabilityPosterior.standard_normal(2)
    information = posterior.information_gain(np.asarray((1.0, 0.0)), 0.1)
    posterior.update(np.asarray((1.0, 0.0)), centered_score=0.5, observation_noise_var=0.1)
    assert information > 0.0
    assert posterior.covariance[0, 0] < 1.0
    assert posterior.covariance[1, 1] == 1.0
    mean, variance = posterior.predict(0.2, np.asarray((1.0, 0.0)), 0.1)
    assert mean > 0.2 and variance > 0.0
