from __future__ import annotations

import numpy as np

from replications.adate_highway_env.adate.importance import (
    challenge_policy,
    defensive_mixture,
    log_importance_weight,
    mixture_of_policies,
    weighted_event_estimate,
)


def test_defensive_support_and_log_weight_are_consistent():
    proposal = defensive_mixture(np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.8, 0.1]), 0.05)
    assert np.all(proposal > 0)
    assert np.isfinite(log_importance_weight(np.array([0.1]), np.array([proposal[0]])))


def test_mixture_of_source_policies_uses_q_times_phi_and_preserves_simplex():
    phi = np.array([0.2, 0.5, 0.3])
    challenges = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    first = challenge_policy(challenges[:, 0], phi)
    second = challenge_policy(challenges[:, 1], phi)
    mixed = mixture_of_policies(challenges, np.array([0.25, 0.75]), phi)
    assert np.allclose(mixed, 0.25 * first + 0.75 * second)
    assert np.isclose(mixed.sum(), 1.0)
    diagnostics = weighted_event_estimate(np.array([1, 0, 1]), np.zeros(3))
    assert diagnostics["mean_weight"] == 1.0
    assert np.isfinite(diagnostics["relative_half_width"])
