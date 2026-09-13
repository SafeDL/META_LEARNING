"""Contracts for leakage-free regression population references."""

from __future__ import annotations

import numpy as np

from mvr.highway.config import RegressionExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.experiments import run_loso_regression_mining as loso
from mvr.highway.diva.regression_reference import build_regression_reference


def test_regression_reference_uses_failure_prevalence_and_beta_smoothing():
    vulnerability = np.array([[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]])
    collisions = np.array([[True, False], [True, False], [True, False]])
    near_misses = np.zeros_like(collisions)

    reference = build_regression_reference(vulnerability, collisions, near_misses)

    assert np.allclose(reference.mean_vulnerability, [0.3, 0.2])
    assert np.allclose(reference.failure_prevalence, [1.0, 0.0])
    assert np.allclose(reference.smoothed_failure_prevalence, [0.8, 0.2])


def test_held_out_target_cannot_change_source_reference_or_prior():
    source_vulnerability = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    )
    source_collisions = np.array(
        [[False, False, True], [False, True, True], [False, False, True]]
    )
    source_near_misses = np.zeros_like(source_collisions)
    held_out_target = np.array([1.0, 1.0, 1.0])

    prior = LowRankPrior.fit(source_vulnerability, rank=2)
    reference = build_regression_reference(
        source_vulnerability, source_collisions, source_near_misses
    )

    assert not np.allclose(prior.mean, held_out_target)
    assert np.allclose(prior.mean, source_vulnerability.mean(axis=0))
    assert np.allclose(reference.failure_prevalence, [0.0, 1 / 3, 1.0])


def test_loso_runner_excludes_target_from_all_population_fitted_quantities(
    monkeypatch,
):
    vulnerability = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )
    collisions = vulnerability >= 0.7
    bank = ResponseBank(
        anchors=np.zeros((3, 2)),
        sut_names=("a", "b", "c"),
        vulnerability=vulnerability,
        collisions=collisions,
        near_misses=np.zeros_like(collisions),
        min_ttc=np.zeros_like(vulnerability),
        min_distance=np.zeros_like(vulnerability),
        completed=np.ones_like(collisions),
    )
    fitted_sources: list[np.ndarray] = []
    reference_sources: list[np.ndarray] = []
    original_fit = loso.LowRankPrior.fit
    original_reference = loso.build_regression_reference

    def record_fit(source: np.ndarray, rank: int) -> LowRankPrior:
        fitted_sources.append(source.copy())
        return original_fit(source, rank)

    def record_reference(
        source: np.ndarray,
        source_collisions: np.ndarray,
        source_near_misses: np.ndarray,
        alpha: float,
        beta: float,
    ):
        reference_sources.append(source.copy())
        return original_reference(
            source, source_collisions, source_near_misses, alpha, beta
        )

    monkeypatch.setattr(loso.LowRankPrior, "fit", record_fit)
    monkeypatch.setattr(loso, "build_regression_reference", record_reference)
    loso.run_loso_regression_mining(
        bank,
        RegressionExperimentConfig(
            num_anchors=3,
            prior_rank=1,
            total_budget=2,
            random_support_repeats=1,
        ),
    )

    for target_index in range(len(bank.sut_names)):
        expected = np.delete(vulnerability, target_index, axis=0)
        assert np.array_equal(fitted_sources[target_index], expected)
        assert np.array_equal(reference_sources[target_index], expected)
