"""Leakage-free DETOUR hierarchy transfer for the isolated Risk Mining fusion chain.

The hierarchy is deliberately a *source-only* ranking prior.  It is useful in
two roles: a K=0 baseline (``detour_static_mining``) and a local-failure
component in posterior-guided Mining mining.  Target vulnerability values are
only read after a support/query index has been selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_sim_env.mining.acquisition import (
    boundary_weight,
    diagnostic_support_indices,
    highest_risk_indices,
)
from highway_sim_env.mining.low_rank_prior import LowRankPrior
from highway_sim_env.mining.mining import MiningTrace, _trace
from highway_sim_env.mining.posterior import adapt_posterior
from replications.detour_highway_env.detour.retrieve import Retriever
from replications.detour_highway_env.detour.selector import prioritize
from replications.detour_highway_env.detour.tree import build_tree


@dataclass(frozen=True)
class DetourHierarchyPrior:
    """A complete DETOUR prioritization order and its normalized rank score."""

    order: np.ndarray
    rank_score: np.ndarray


def hierarchy_prior(
    history_features: np.ndarray,
    candidate_features: np.ndarray,
    failed_history: np.ndarray,
    seed: int,
) -> DetourHierarchyPrior:
    """Build the original Ward/retrieval hierarchy from labelled source history.

    ``failed_history`` must come from source SUTs only.  Candidate features are
    unlabelled logical-scenario inputs, so this function cannot inspect target
    outcomes by construction.
    """
    candidates = np.asarray(candidate_features, dtype=float)
    if candidates.ndim != 2 or len(candidates) < 1:
        raise ValueError("candidate_features must be a non-empty two-dimensional array")
    tree = build_tree(history_features, candidates, failed_history)
    order, _traces, _decisions = prioritize(Retriever(tree, int(seed)), len(candidates))
    ranking = np.asarray(order, dtype=int)
    if len(ranking) != len(candidates) or len(np.unique(ranking)) != len(candidates):
        raise RuntimeError("DETOUR hierarchy did not return a complete unique ranking")
    rank_score = np.empty(len(candidates), dtype=float)
    # A rank score is intentionally ordinal: DETOUR exposes a prioritization,
    # not a calibrated target failure probability.
    rank_score[ranking] = np.linspace(1.0, 0.0, len(candidates), endpoint=True)
    return DetourHierarchyPrior(ranking, rank_score)


def detour_static_mining(
    hierarchy: DetourHierarchyPrior,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    budget: int,
) -> MiningTrace:
    """K=0 historical-neighbourhood transfer baseline under a fixed budget."""
    if not 0 < budget <= len(hierarchy.order):
        raise ValueError("budget must be in [1, number of candidates]")
    return _trace(
        "DETOUR Hierarchy (K=0)", hierarchy.order[:budget], collisions, near_misses
    )


def hierarchy_aware_support_indices(
    prior: LowRankPrior,
    hierarchy: DetourHierarchyPrior,
    target_vulnerability: np.ndarray,
    count: int,
    hierarchy_weight: float = 0.25,
    observation_noise: float = 0.03,
) -> np.ndarray:
    """Choose diagnostic probes with boundary information and source locality.

    DETOUR does not make a target label visible here.  Its rank only down/up
    weights otherwise diagnostic probes toward source-failure neighbourhoods;
    the newly revealed support value is then the sole target-side input to the
    next posterior update.
    """
    if not 0.0 <= hierarchy_weight <= 1.0:
        raise ValueError("hierarchy_weight must be in [0, 1]")
    target = np.asarray(target_vulnerability, dtype=float)
    if target.shape != prior.mean.shape or hierarchy.rank_score.shape != target.shape:
        raise ValueError("prior, hierarchy, and target vulnerability must align")
    if not 0 <= count <= len(target):
        raise ValueError("count must be in [0, number of candidates]")
    selected: list[int] = []
    posterior = adapt_posterior(
        prior, np.asarray([], dtype=int), np.asarray([], dtype=float), observation_noise
    )
    for _ in range(count):
        candidates = np.asarray(
            [index for index in range(len(target)) if index not in selected], dtype=int
        )
        basis = prior.basis[candidates]
        information_gain = np.einsum("ij,jk,ik->i", basis, posterior.covariance, basis)
        diagnostic_score = information_gain * boundary_weight(
            posterior.prediction[candidates]
        )
        locality = (1.0 - hierarchy_weight) + hierarchy_weight * hierarchy.rank_score[candidates]
        scores = diagnostic_score * locality
        order = np.lexsort((candidates, -scores))
        chosen = int(candidates[order[0]])
        selected.append(chosen)
        support = np.asarray(selected, dtype=int)
        posterior = adapt_posterior(
            prior, support, target[support], observation_noise
        )
    return np.asarray(selected, dtype=int)


def detour_fused_diagnostic_mining(
    prior: LowRankPrior,
    hierarchy: DetourHierarchyPrior,
    vulnerability: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    support_budget: int,
    total_budget: int,
    detour_weight: float = 0.10,
) -> MiningTrace:
    """Diagnose the target first, then blend posterior risk with DETOUR locality.

    The blend is used only after the K diagnostic reveals.  This prevents the
    historical hierarchy from masquerading as target adaptation and makes the
    method falsifiable against both the K=0 DETOUR baseline and Mining without
    hierarchy transfer.
    """
    if not 0.0 <= detour_weight <= 1.0:
        raise ValueError("detour_weight must be in [0, 1]")
    target = np.asarray(vulnerability, dtype=float)
    if target.shape != prior.mean.shape or hierarchy.rank_score.shape != target.shape:
        raise ValueError("prior, hierarchy, and target vulnerability must align")
    if not 0 < support_budget < total_budget <= len(target):
        raise ValueError("require 0 < support_budget < total_budget <= candidates")
    support = diagnostic_support_indices(prior, target, support_budget)
    posterior = adapt_posterior(prior, support, target[support])
    posterior_score = np.clip(posterior.prediction, 0.0, 1.0)
    fused_score = (
        (1.0 - detour_weight) * posterior_score
        + detour_weight * hierarchy.rank_score
    )
    mining = highest_risk_indices(fused_score, total_budget - len(support), support)
    queried = np.concatenate((support, mining))
    return _trace(
        "Risk Mining + DETOUR Hierarchy",
        queried,
        collisions,
        near_misses,
    )


def detour_guided_diagnostic_mining(
    prior: LowRankPrior,
    hierarchy: DetourHierarchyPrior,
    vulnerability: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    support_budget: int,
    total_budget: int,
    hierarchy_weight: float = 0.10,
) -> MiningTrace:
    """Use DETOUR only during K-shot probe design, then mine posterior risk."""
    target = np.asarray(vulnerability, dtype=float)
    if not 0 < support_budget < total_budget <= len(target):
        raise ValueError("require 0 < support_budget < total_budget <= candidates")
    support = hierarchy_aware_support_indices(
        prior, hierarchy, target, support_budget, hierarchy_weight
    )
    posterior = adapt_posterior(prior, support, target[support])
    mining = highest_risk_indices(
        posterior.prediction, total_budget - len(support), support
    )
    return _trace(
        "DETOUR-Guided Risk Diagnosis",
        np.concatenate((support, mining)),
        collisions,
        near_misses,
    )
