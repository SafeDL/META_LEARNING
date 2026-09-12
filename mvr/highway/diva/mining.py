"""Fixed-budget methods evaluated against an already generated response bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvr.highway.diva.acquisition import diagnostic_support_indices, highest_risk_indices
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import adapt_posterior


@dataclass(frozen=True)
class MiningTrace:
    """Every target reveal, including diagnostic support, under one budget."""

    method: str
    queried_indices: np.ndarray
    cumulative_critical_score: np.ndarray
    collision_count: int
    failure_count: int

    @property
    def critical_score(self) -> float:
        return float(self.cumulative_critical_score[-1])


def critical_rewards(collisions: np.ndarray, near_misses: np.ndarray) -> np.ndarray:
    """Score collision as one and an otherwise near-miss as one half."""
    return np.where(collisions, 1.0, np.where(near_misses, 0.5, 0.0))


def _trace(
    method: str,
    queried: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
) -> MiningTrace:
    if len(np.unique(queried)) != len(queried):
        raise ValueError("A scenario may be queried only once")
    rewards = critical_rewards(collisions[queried], near_misses[queried])
    return MiningTrace(
        method=method,
        queried_indices=queried,
        cumulative_critical_score=np.cumsum(rewards),
        collision_count=int(collisions[queried].sum()),
        failure_count=int((collisions[queried] | near_misses[queried]).sum()),
    )


def random_mining(
    collisions: np.ndarray,
    near_misses: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> MiningTrace:
    queried = rng.choice(len(collisions), size=budget, replace=False)
    return _trace("Random", queried, collisions, near_misses)


def shared_prior_mining(
    prior: LowRankPrior, collisions: np.ndarray, near_misses: np.ndarray, budget: int
) -> MiningTrace:
    queried = highest_risk_indices(prior.mean, budget)
    return _trace("Shared Prior", queried, collisions, near_misses)


def adapted_mining(
    method: str,
    prior: LowRankPrior,
    vulnerability: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    support_indices: np.ndarray,
    budget: int,
) -> MiningTrace:
    """Reveal K support outcomes, then mine the remaining budget once."""
    support = np.asarray(support_indices, dtype=int)
    if len(support) >= budget:
        raise ValueError("support budget must be smaller than total budget")
    posterior = adapt_posterior(prior, support, vulnerability[support])
    queries = np.concatenate(
        [
            support,
            highest_risk_indices(posterior.prediction, budget - len(support), support),
        ]
    )
    return _trace(method, queries, collisions, near_misses)


def highest_risk_support_mining(
    prior: LowRankPrior,
    vulnerability: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    support_budget: int,
    total_budget: int,
) -> MiningTrace:
    """Adapt after probing the K highest-risk scenarios under the shared prior."""
    support = highest_risk_indices(prior.mean, support_budget)
    return adapted_mining(
        "Highest-Risk Support + Adaptation",
        prior,
        vulnerability,
        collisions,
        near_misses,
        support,
        total_budget,
    )


def diagnostic_mining(
    prior: LowRankPrior,
    vulnerability: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    support_budget: int,
    total_budget: int,
) -> MiningTrace:
    support = diagnostic_support_indices(prior, vulnerability, support_budget)
    return adapted_mining(
        "DIVA Diagnostic + Adaptation",
        prior,
        vulnerability,
        collisions,
        near_misses,
        support,
        total_budget,
    )
