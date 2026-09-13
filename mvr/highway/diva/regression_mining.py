"""Sequential population-referenced regression-vulnerability discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.outcomes import formal_rewards
from mvr.highway.diva.posterior import LatentPosterior, adapt_posterior
from mvr.highway.diva.regression_probability import critical_probability
from mvr.highway.diva.regression_reference import RegressionReference


@dataclass(frozen=True)
class RegressionMiningTrace:
    """The ordered selections and cumulative outcomes from one target replay."""

    queried_indices: np.ndarray
    raw_critical_curve: np.ndarray
    regression_critical_curve: np.ndarray
    target_specific_failure_curve: np.ndarray


def regression_scores(
    prior: LowRankPrior,
    posterior: LatentPosterior,
    reference: RegressionReference,
    threshold: float = 0.75,
) -> np.ndarray:
    """Score anchors by posterior critical probability times historical rarity."""
    probabilities = critical_probability(prior, posterior, threshold)
    if reference.smoothed_failure_prevalence.shape != probabilities.shape:
        raise ValueError("reference must contain one prevalence per anchor")
    return probabilities * (1.0 - reference.smoothed_failure_prevalence)


def select_next_regression_index(scores: np.ndarray, selected: list[int]) -> int:
    """Choose the highest untested anchor with deterministic lower-index ties."""
    values = np.asarray(scores, dtype=float)
    blocked = set(selected)
    candidates = np.asarray(
        [index for index in range(len(values)) if index not in blocked], dtype=int
    )
    if not len(candidates):
        raise ValueError("No untested anchors remain")
    order = np.lexsort((candidates, -values[candidates]))
    return int(candidates[order[0]])


def run_regression_mining(
    prior: LowRankPrior,
    reference: RegressionReference,
    target_vulnerability: np.ndarray,
    target_collisions: np.ndarray,
    target_near_misses: np.ndarray,
    total_budget: int,
    threshold: float = 0.75,
) -> RegressionMiningTrace:
    """Select, reveal one target outcome, update, and repeat for the full budget."""
    vulnerability = np.asarray(target_vulnerability, dtype=float)
    collisions = np.asarray(target_collisions, dtype=bool)
    near_misses = np.asarray(target_near_misses, dtype=bool)
    if vulnerability.shape != prior.mean.shape:
        raise ValueError("target_vulnerability must contain one value per anchor")
    if (
        collisions.shape != vulnerability.shape
        or near_misses.shape != vulnerability.shape
    ):
        raise ValueError("target outcome arrays must match target_vulnerability")
    if not 0 < total_budget <= len(vulnerability):
        raise ValueError("total_budget must be in [1, number of anchors]")

    selected: list[int] = []
    observations: list[float] = []
    posterior = adapt_posterior(prior, np.asarray([], dtype=int), np.asarray([]))
    raw_total = regression_total = target_specific_total = 0.0
    raw_curve: list[float] = []
    regression_curve: list[float] = []
    target_specific_curve: list[float] = []
    for _ in range(total_budget):
        scores = regression_scores(prior, posterior, reference, threshold)
        chosen = select_next_regression_index(scores, selected)

        # Target values are accessed only after this selection is finalized.
        selected.append(chosen)
        observation = float(vulnerability[chosen])
        observations.append(observation)
        reward = float(formal_rewards(collisions[chosen], near_misses[chosen]))
        raw_total += reward
        regression_total += reward * (
            1.0 - reference.smoothed_failure_prevalence[chosen]
        )
        target_specific_total += float(
            reward > 0.0 and reference.failure_prevalence[chosen] <= 0.20
        )
        raw_curve.append(raw_total)
        regression_curve.append(regression_total)
        target_specific_curve.append(target_specific_total)
        posterior = adapt_posterior(
            prior,
            np.asarray(selected, dtype=int),
            np.asarray(observations, dtype=float),
        )

    return RegressionMiningTrace(
        queried_indices=np.asarray(selected, dtype=int),
        raw_critical_curve=np.asarray(raw_curve),
        regression_critical_curve=np.asarray(regression_curve),
        target_specific_failure_curve=np.asarray(target_specific_curve),
    )
