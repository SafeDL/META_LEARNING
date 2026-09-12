from __future__ import annotations

import numpy as np

from mvr.highway.diva.acquisition import diagnostic_support_indices
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.mining import adapted_mining, diagnostic_mining
from mvr.highway.diva.posterior import adapt_posterior


def test_prior_uses_only_source_rows_and_posterior_uses_only_support():
    source = np.array([[0.1, 0.4, 0.3, 0.6], [0.2, 0.5, 0.2, 0.7], [0.3, 0.6, 0.1, 0.8]])
    target = np.array([0.9, 0.8, 0.1, 0.2])
    prior = LowRankPrior.fit(source, rank=2)
    posterior = adapt_posterior(prior, np.array([0, 1]), target[:2])
    changed_target = target.copy()
    changed_target[2:] = 99.0
    unchanged_posterior = adapt_posterior(prior, np.array([0, 1]), changed_target[:2])
    assert np.allclose(posterior.prediction, unchanged_posterior.prediction)
    assert not np.allclose(prior.mean, target)


def test_diagnostic_and_adapted_mining_count_support_in_fixed_budget():
    prior = LowRankPrior.fit(
        np.array([[0.1, 0.5, 0.2, 0.9, 0.4], [0.2, 0.4, 0.3, 0.8, 0.5]]),
        rank=2,
    )
    vulnerability = np.array([0.8, 0.7, 0.1, 0.9, 0.2])
    collisions = np.array([True, False, False, True, False])
    near_misses = np.array([False, True, False, False, False])
    support = diagnostic_support_indices(prior, 2)
    trace = adapted_mining(
        "Random Support + Adaptation",
        prior,
        vulnerability,
        collisions,
        near_misses,
        support,
        4,
    )
    diagnostic = diagnostic_mining(
        prior, vulnerability, collisions, near_misses, 2, 4
    )
    assert len(trace.queried_indices) == len(diagnostic.queried_indices) == 4
    assert len(np.unique(trace.queried_indices)) == 4
