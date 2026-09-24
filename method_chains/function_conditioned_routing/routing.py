"""Function-aware AdaTE response routing with a Mining fallback.

Only responses at indices already selected by the method enter any fit.  The
complete target vector is accepted by the experiment driver as an oracle and
is indexed immediately after each query, mirroring a real target execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env_benchmark.mining.low_rank_prior import LowRankPrior
from highway_env_benchmark.mining.mining import MiningTrace, _trace
from highway_env_benchmark.mining.posterior import adapt_posterior
from replications.adate_highway_env.adate.mixture import (
    simplex_least_squares,
    uniform_alpha,
)

from .config import RoutingExperimentConfig


@dataclass(frozen=True)
class RoutingPrediction:
    """Auditable prediction assembled from revealed target responses only."""

    prediction: np.ndarray
    mining_prediction: np.ndarray
    routed_prediction: np.ndarray
    global_alpha: np.ndarray
    mode_alphas: dict[str, np.ndarray]
    mode_gates: dict[str, float]
    revealed_rmse: float


@dataclass(frozen=True)
class RoutedMiningResult:
    """Mining trace plus the state immediately after diagnostic support."""

    trace: MiningTrace
    support_indices: np.ndarray
    support_prediction: RoutingPrediction


def _fit_routing_weights(
    source_responses: np.ndarray,
    modes: np.ndarray,
    selected: np.ndarray,
    revealed: np.ndarray,
    local_shrinkage: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model_count = source_responses.shape[0]
    if not len(selected):
        global_alpha = uniform_alpha(model_count)
    else:
        global_alpha = simplex_least_squares(
            source_responses[:, selected].T,
            revealed,
        ).alpha
    mode_alphas: dict[str, np.ndarray] = {}
    for mode in np.unique(modes):
        local_mask = modes[selected] == mode if len(selected) else np.zeros(0, dtype=bool)
        local_indices = selected[local_mask]
        if not len(local_indices):
            mode_alphas[str(mode)] = global_alpha.copy()
            continue
        local_alpha = simplex_least_squares(
            source_responses[:, local_indices].T,
            revealed[local_mask],
            previous=global_alpha,
        ).alpha
        reliability = len(local_indices) / (len(local_indices) + local_shrinkage)
        mode_alphas[str(mode)] = (
            reliability * local_alpha + (1.0 - reliability) * global_alpha
        )
    return global_alpha, mode_alphas


def routing_prediction(
    source_responses: np.ndarray,
    modes: np.ndarray,
    prior: LowRankPrior,
    selected: np.ndarray,
    revealed: np.ndarray,
    config: RoutingExperimentConfig,
    *,
    functional: bool = True,
    trusted: bool = True,
) -> RoutingPrediction:
    """Predict every candidate without consulting unqueried target outcomes."""
    sources = np.asarray(source_responses, dtype=float)
    labels = np.asarray(modes, dtype=str)
    indices = np.asarray(selected, dtype=int)
    outcomes = np.asarray(revealed, dtype=float)
    if sources.ndim != 2 or sources.shape[1] != len(labels):
        raise ValueError("source responses and modes must share the candidate axis")
    if outcomes.shape != (len(indices),):
        raise ValueError("one revealed response is required per selected index")
    global_alpha, mode_alphas = _fit_routing_weights(
        sources,
        labels,
        indices,
        outcomes,
        config.local_shrinkage,
    )
    routed = np.empty(sources.shape[1], dtype=float)
    for mode in np.unique(labels):
        alpha = mode_alphas[str(mode)] if functional else global_alpha
        routed[labels == mode] = alpha @ sources[:, labels == mode]
    posterior = adapt_posterior(prior, indices, outcomes)
    mining = np.clip(posterior.prediction, 0.0, 1.0)
    gates: dict[str, float] = {}
    for mode in np.unique(labels):
        local_mask = labels[indices] == mode if len(indices) else np.zeros(0, dtype=bool)
        local_indices = indices[local_mask]
        if not trusted:
            gate = 1.0
        elif not len(local_indices):
            gate = 0.0
        else:
            residual = routed[local_indices] - outcomes[local_mask]
            rmse = float(np.sqrt(np.mean(residual**2)))
            gate = np.exp(-(rmse / config.gate_error_scale) ** 2)
        gates[str(mode)] = float(np.clip(gate, 0.0, 1.0))
    gate_vector = np.asarray([gates[str(mode)] for mode in labels])
    prediction = np.clip(gate_vector * routed + (1.0 - gate_vector) * mining, 0.0, 1.0)
    revealed_rmse = (
        float(np.sqrt(np.mean((prediction[indices] - outcomes) ** 2)))
        if len(indices)
        else 0.0
    )
    return RoutingPrediction(
        prediction,
        mining,
        routed,
        global_alpha,
        mode_alphas,
        gates,
        revealed_rmse,
    )


def adate_support(
    source_responses: np.ndarray,
    target_oracle: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Use the validated A0 rule so diagnosis preserves high-risk discovery."""
    sources = np.asarray(source_responses, dtype=float)
    truth = np.asarray(target_oracle, dtype=float)
    selected: list[int] = []
    revealed: list[float] = []
    alpha = uniform_alpha(sources.shape[0])
    for _ in range(budget):
        scores = alpha @ sources
        if selected:
            scores[np.asarray(selected, dtype=int)] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        revealed.append(float(truth[chosen]))
        alpha = simplex_least_squares(
            sources[:, selected].T,
            np.asarray(revealed),
            previous=alpha,
        ).alpha
    return np.asarray(selected, dtype=int)


def coverage_adate_support(
    source_responses: np.ndarray,
    modes: np.ndarray,
    target_oracle: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Preserve AdaTE risk ordering while guaranteeing one probe per function."""
    sources = np.asarray(source_responses, dtype=float)
    labels = np.asarray(modes, dtype=str)
    truth = np.asarray(target_oracle, dtype=float)
    unique_modes = tuple(dict.fromkeys(labels))
    if budget < len(unique_modes):
        raise ValueError("budget must cover every functional mode")
    selected: list[int] = []
    revealed: list[float] = []
    alpha = uniform_alpha(sources.shape[0])
    for step in range(budget):
        scores = alpha @ sources
        if selected:
            scores[np.asarray(selected, dtype=int)] = -np.inf
        counts = {
            mode: int(np.sum(labels[np.asarray(selected, dtype=int)] == mode))
            for mode in unique_modes
        }
        missing = tuple(mode for mode in unique_modes if counts[mode] == 0)
        remaining = budget - step
        if missing and remaining <= len(missing):
            allowed = np.isin(labels, missing)
            scores[~allowed] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        revealed.append(float(truth[chosen]))
        alpha = simplex_least_squares(
            sources[:, selected].T,
            np.asarray(revealed),
            previous=alpha,
        ).alpha
    return np.asarray(selected, dtype=int)


def _highest_unqueried(scores: np.ndarray, queried: list[int]) -> int:
    candidates = np.asarray(
        [index for index in range(len(scores)) if index not in queried],
        dtype=int,
    )
    order = np.lexsort((candidates, -np.asarray(scores)[candidates]))
    return int(candidates[order[0]])


def functional_routed_mining(
    method: str,
    source_responses: np.ndarray,
    modes: np.ndarray,
    prior: LowRankPrior,
    target_oracle: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    config: RoutingExperimentConfig,
    *,
    functional: bool = True,
    trusted: bool = True,
    support_policy: str = "coverage",
) -> RoutedMiningResult:
    """Diagnose once, then update routing after every target execution."""
    if support_policy == "adate":
        support = adate_support(source_responses, target_oracle, config.support_budget)
    elif support_policy == "coverage":
        support = coverage_adate_support(
            source_responses,
            modes,
            target_oracle,
            config.support_budget,
        )
    else:
        raise ValueError("support_policy must be 'adate' or 'coverage'")
    queried = support.tolist()
    revealed = np.asarray(target_oracle, dtype=float)[support].tolist()
    support_state = routing_prediction(
        source_responses,
        modes,
        prior,
        support,
        np.asarray(revealed),
        config,
        functional=functional,
        trusted=trusted,
    )
    while len(queried) < config.total_budget:
        state = routing_prediction(
            source_responses,
            modes,
            prior,
            np.asarray(queried, dtype=int),
            np.asarray(revealed),
            config,
            functional=functional,
            trusted=trusted,
        )
        chosen = _highest_unqueried(state.prediction, queried)
        queried.append(chosen)
        revealed.append(float(target_oracle[chosen]))
    trace = _trace(
        method,
        np.asarray(queried, dtype=int),
        np.asarray(collisions, dtype=bool),
        np.asarray(near_misses, dtype=bool),
    )
    return RoutedMiningResult(trace, support, support_state)


def sequential_mining_mining(
    method: str,
    prior: LowRankPrior,
    target_oracle: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    support: np.ndarray,
    budget: int,
) -> MiningTrace:
    """Strong baseline that updates the ordinary Mining posterior after every query."""
    queried = np.asarray(support, dtype=int).tolist()
    revealed = np.asarray(target_oracle, dtype=float)[support].tolist()
    while len(queried) < budget:
        posterior = adapt_posterior(
            prior,
            np.asarray(queried, dtype=int),
            np.asarray(revealed),
        )
        chosen = _highest_unqueried(posterior.prediction, queried)
        queried.append(chosen)
        revealed.append(float(target_oracle[chosen]))
    return _trace(
        method,
        np.asarray(queried, dtype=int),
        collisions,
        near_misses,
    )
