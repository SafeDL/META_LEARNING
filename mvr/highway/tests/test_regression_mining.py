"""Behavioral contracts for sequential PR-BRVT mining."""

from __future__ import annotations

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import LatentPosterior
from mvr.highway.diva.regression_mining import (
    regression_scores,
    run_regression_mining,
)
from mvr.highway.diva.regression_probability import critical_probability
from mvr.highway.diva.regression_reference import RegressionReference


def _prior() -> LowRankPrior:
    return LowRankPrior(
        mean=np.full(3, 0.75),
        basis=np.array([[1.0, 0.0], [0.0, 0.5], [0.7, 0.7]]),
        latent_covariance=np.eye(2),
        explained_variance_ratio=np.array([0.7, 0.3]),
    )


def _reference() -> RegressionReference:
    return RegressionReference(
        mean_vulnerability=np.full(3, 0.75),
        failure_prevalence=np.zeros(3),
        smoothed_failure_prevalence=np.zeros(3),
    )


def test_critical_probability_increases_with_posterior_mean():
    prior = LowRankPrior(
        mean=np.zeros(1),
        basis=np.ones((1, 1)),
        latent_covariance=np.ones((1, 1)),
        explained_variance_ratio=np.ones(1),
    )
    low = LatentPosterior(np.array([0.5]), np.eye(1) * 0.01, np.array([0.5]))
    high = LatentPosterior(np.array([0.9]), np.eye(1) * 0.01, np.array([0.9]))

    assert critical_probability(prior, high)[0] > critical_probability(prior, low)[0]


def test_uncertainty_changes_probability_near_the_critical_threshold():
    prior = LowRankPrior(
        mean=np.array([0.70]),
        basis=np.ones((1, 1)),
        latent_covariance=np.eye(1),
        explained_variance_ratio=np.ones(1),
    )
    low_uncertainty = LatentPosterior(
        np.zeros(1), np.eye(1) * 0.0001, np.array([0.70])
    )
    high_uncertainty = LatentPosterior(
        np.zeros(1), np.eye(1), np.array([0.70])
    )

    assert (
        critical_probability(prior, high_uncertainty)[0]
        > critical_probability(prior, low_uncertainty)[0]
    )


def test_regression_ranking_prefers_target_risk_that_is_historically_rare():
    prior = LowRankPrior(
        mean=np.zeros(2),
        basis=np.eye(2),
        latent_covariance=np.eye(2),
        explained_variance_ratio=np.array([0.5, 0.5]),
    )
    posterior = LatentPosterior(
        mean=np.zeros(2), covariance=np.eye(2), prediction=np.array([0.9, 0.8])
    )
    reference = RegressionReference(
        mean_vulnerability=np.zeros(2),
        failure_prevalence=np.array([0.9, 0.1]),
        smoothed_failure_prevalence=np.array([0.9, 0.1]),
    )

    scores = regression_scores(prior, posterior, reference)

    assert scores[1] > scores[0]


def test_sequential_reveal_changes_only_the_next_regression_selection():
    prior = _prior()
    reference = _reference()
    collisions = np.zeros(3, dtype=bool)
    near_misses = np.zeros(3, dtype=bool)
    first = run_regression_mining(
        prior, reference, np.full(3, 0.75), collisions, near_misses, 1
    ).queried_indices[0]
    low_observation = np.full(3, 0.75)
    high_observation = low_observation.copy()
    low_observation[first] = 0.0
    high_observation[first] = 1.0

    low_trace = run_regression_mining(
        prior, reference, low_observation, collisions, near_misses, 2
    )
    high_trace = run_regression_mining(
        prior, reference, high_observation, collisions, near_misses, 2
    )

    assert low_trace.queried_indices[0] == high_trace.queried_indices[0] == first
    assert low_trace.queried_indices[1] != high_trace.queried_indices[1]


def test_regression_mining_never_repeats_queries_and_tracks_required_metrics():
    trace = run_regression_mining(
        _prior(),
        _reference(),
        np.array([1.0, 0.2, 0.8]),
        np.array([True, False, False]),
        np.array([False, True, False]),
        3,
    )

    assert len(np.unique(trace.queried_indices)) == len(trace.queried_indices)
    assert trace.raw_critical_curve[-1] == 1.5
    assert trace.regression_critical_curve[-1] == 1.5
    assert trace.target_specific_failure_curve[-1] == 2.0
