from __future__ import annotations

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.diva.differential_acquisition import differential_acquisition
from mvr.highway.diva.differential_mining import run_differential_mining
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.population_reference import PopulationReference
from mvr.highway.diva.posterior import LatentPosterior


def _prior() -> LowRankPrior:
    return LowRankPrior(
        mean=np.full(3, 0.75),
        basis=np.array([[1.0, 0.0], [0.0, 0.5], [0.7, 0.7]]),
        latent_covariance=np.eye(2),
        explained_variance_ratio=np.array([0.7, 0.3]),
    )


def _population() -> PopulationReference:
    return PopulationReference(
        mean_vulnerability=np.full(3, 0.75),
        failure_prevalence=np.zeros(3),
        smoothed_failure_prevalence=np.zeros(3),
    )


def test_differential_acquisition_prioritizes_a_novel_target_risk():
    prior = LowRankPrior(
        mean=np.zeros(2),
        basis=np.eye(2),
        latent_covariance=np.eye(2),
        explained_variance_ratio=np.array([0.5, 0.5]),
    )
    posterior = LatentPosterior(
        mean=np.zeros(2), covariance=np.zeros((2, 2)), prediction=np.array([0.9, 0.8])
    )
    population = PopulationReference(
        mean_vulnerability=np.zeros(2),
        failure_prevalence=np.array([1.0, 0.0]),
        smoothed_failure_prevalence=np.array([1.0, 0.0]),
    )

    scores = differential_acquisition(prior, posterior, population)

    assert scores[1] > scores[0]


def test_sequential_reveal_changes_only_future_acquisition_decisions():
    prior = _prior()
    population = _population()
    collisions = np.zeros(3, dtype=bool)
    near_misses = np.zeros(3, dtype=bool)
    first_only = run_differential_mining(
        prior, population, np.full(3, 0.75), collisions, near_misses, 1
    )
    first = first_only.queried_indices[0]
    low_observation = np.full(3, 0.75)
    high_observation = low_observation.copy()
    low_observation[first] = 0.0
    high_observation[first] = 1.0

    low_trace = run_differential_mining(
        prior, population, low_observation, collisions, near_misses, 2
    )
    high_trace = run_differential_mining(
        prior, population, high_observation, collisions, near_misses, 2
    )

    assert low_trace.queried_indices[0] == high_trace.queried_indices[0] == first
    assert low_trace.queried_indices[1] != high_trace.queried_indices[1]


def test_differential_mining_never_repeats_a_query_and_tracks_all_metrics():
    trace = run_differential_mining(
        _prior(),
        _population(),
        np.array([1.0, 0.2, 0.8]),
        np.array([True, False, False]),
        np.array([False, True, False]),
        3,
    )

    assert len(np.unique(trace.queried_indices)) == len(trace.queried_indices)
    assert trace.raw_critical_curve[-1] == 1.5
    assert trace.differential_critical_curve[-1] == 1.5
    assert trace.target_specific_failure_curve[-1] == 2.0


def test_pr_dvm_configuration_does_not_require_a_legacy_support_budget():
    ExperimentConfig(support_budget=20).validate()
