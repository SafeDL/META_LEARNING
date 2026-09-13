from __future__ import annotations

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.population_reference import build_population_reference


def test_population_reference_reports_unsmoothed_and_beta_smoothed_prevalence():
    vulnerability = np.array(
        [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]], dtype=float
    )
    collisions = np.array([[True, False], [True, False], [True, False]])
    near_misses = np.zeros_like(collisions)

    reference = build_population_reference(vulnerability, collisions, near_misses)

    assert np.allclose(reference.mean_vulnerability, [0.3, 0.2])
    assert np.allclose(reference.failure_prevalence, [1.0, 0.0])
    assert np.allclose(reference.smoothed_failure_prevalence, [0.8, 0.2])


def test_loso_reference_and_prior_exclude_the_held_out_target_row():
    source_vulnerability = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    )
    source_collisions = np.array(
        [[False, False, True], [False, True, True], [False, False, True]]
    )
    source_near_misses = np.zeros_like(source_collisions)
    held_out_target = np.array([1.0, 1.0, 1.0])

    prior = LowRankPrior.fit(source_vulnerability, rank=2)
    reference = build_population_reference(
        source_vulnerability, source_collisions, source_near_misses
    )
    changed_target = np.zeros_like(held_out_target)

    assert not np.allclose(prior.mean, held_out_target)
    assert np.allclose(prior.mean, source_vulnerability.mean(axis=0))
    assert np.allclose(reference.failure_prevalence, [0.0, 1 / 3, 1.0])
    assert not np.allclose(source_vulnerability.mean(axis=0), changed_target)
    repeated_prior = LowRankPrior.fit(source_vulnerability, 2)
    assert np.allclose(prior.latent_covariance, repeated_prior.latent_covariance)
