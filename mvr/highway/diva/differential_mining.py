"""Leakage-free sequential PR-DVM replay over a complete response bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvr.highway.diva.differential_acquisition import select_next_differential_index
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.outcomes import formal_rewards
from mvr.highway.diva.population_reference import PopulationReference
from mvr.highway.diva.posterior import adapt_posterior


@dataclass(frozen=True)
class DifferentialMiningTrace:
    """All target outcomes revealed during one PR-DVM budgeted replay."""

    queried_indices: np.ndarray
    raw_critical_curve: np.ndarray
    differential_critical_curve: np.ndarray
    target_specific_failure_curve: np.ndarray


def run_differential_mining(
    prior: LowRankPrior,
    population: PopulationReference,
    target_vulnerability: np.ndarray,
    target_collisions: np.ndarray,
    target_near_misses: np.ndarray,
    total_budget: int,
    differential_weight: float = 1.0,
    shared_weight: float = 0.10,
    uncertainty_weight: float = 0.25,
) -> DifferentialMiningTrace:
    """Select, reveal, update, and repeat without looking ahead at target data."""
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
    posterior = adapt_posterior(
        prior, np.asarray([], dtype=int), np.asarray([], dtype=float)
    )
    raw_curve: list[float] = []
    differential_curve: list[float] = []
    target_specific_curve: list[float] = []
    raw_total = differential_total = target_specific_total = 0.0
    for _ in range(total_budget):
        chosen = select_next_differential_index(
            prior,
            posterior,
            population,
            selected,
            differential_weight,
            shared_weight,
            uncertainty_weight,
        )
        selected.append(chosen)
        observations.append(float(vulnerability[chosen]))

        raw_reward = float(formal_rewards(collisions[chosen], near_misses[chosen]))
        raw_total += raw_reward
        differential_total += raw_reward * (
            1.0 - population.smoothed_failure_prevalence[chosen]
        )
        target_specific_total += float(
            raw_reward > 0.0 and population.failure_prevalence[chosen] <= 0.20
        )
        raw_curve.append(raw_total)
        differential_curve.append(differential_total)
        target_specific_curve.append(target_specific_total)

        posterior = adapt_posterior(
            prior,
            np.asarray(selected, dtype=int),
            np.asarray(observations, dtype=float),
        )

    return DifferentialMiningTrace(
        queried_indices=np.asarray(selected, dtype=int),
        raw_critical_curve=np.asarray(raw_curve),
        differential_critical_curve=np.asarray(differential_curve),
        target_specific_failure_curve=np.asarray(target_specific_curve),
    )
