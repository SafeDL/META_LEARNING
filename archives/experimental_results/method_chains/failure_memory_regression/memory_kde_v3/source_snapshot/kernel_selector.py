"""Locally weighted evidence selector for failure-memory replay experiments.

This is a new selector version. `selector_v2` remains frozen because its source
hash is part of the completed physical holdout protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from method_chains.failure_memory_regression.pattern_memory import active_values
from method_chains.failure_memory_regression.replay_utils import (
    contextual_history, is_usable_outcome, task_reward,
)
from method_chains.failure_memory_regression.selector_v2 import (
    TargetOracle, _art_choice,
)


@dataclass(frozen=True)
class KernelMemoryConfig:
    bandwidth: float = 0.18
    source_weight: float = 0.7
    source_support_scale: float = 1.0
    prior_failure_probability: float = 0.1
    prior_strength: float = 3.0
    exploration: float = 0.1
    neighbors: int = 8

    def __post_init__(self) -> None:
        if self.bandwidth <= 0 or self.source_support_scale <= 0:
            raise ValueError("kernel bandwidth and source support scale must be positive")
        if not 0 <= self.source_weight <= 1:
            raise ValueError("source weight must be in [0, 1]")
        if not 0 < self.prior_failure_probability < 1 or self.prior_strength <= 0:
            raise ValueError("invalid beta prior")
        if self.exploration < 0 or self.neighbors < 1:
            raise ValueError("exploration must be nonnegative and neighbors positive")


def _coordinates(items: list[dict]) -> np.ndarray:
    if not items:
        return np.empty((0, 0), dtype=float)
    return np.asarray([active_values(item.get("scenario", item)) for item in items],
                      dtype=float)


def _local_evidence(candidate: np.ndarray, points: np.ndarray,
                    labels: np.ndarray, config: KernelMemoryConfig,
                    family_weights: np.ndarray | None = None) -> tuple[float, float]:
    if not len(points):
        return config.prior_failure_probability, 0.0
    distances = np.linalg.norm(points - candidate, axis=1) / np.sqrt(points.shape[1])
    count = min(config.neighbors, len(points))
    nearest = np.argpartition(distances, count - 1)[:count]
    weights = np.exp(-0.5 * (distances[nearest] / config.bandwidth) ** 2)
    if family_weights is not None:
        weights *= family_weights[nearest]
    support = float(weights.sum())
    alpha = (config.prior_failure_probability * config.prior_strength +
             float(weights @ labels[nearest]))
    beta = ((1.0 - config.prior_failure_probability) * config.prior_strength +
            float(weights @ (1.0 - labels[nearest])))
    return alpha / (alpha + beta), support


def _source_family_weights(history: list[dict]) -> np.ndarray:
    if not history:
        return np.empty(0, dtype=float)
    family_by_row = [str(row.get("family", row.get("build_id", "unknown")))
                     for row in history]
    counts: dict[str, int] = {}
    for family in family_by_row:
        counts[family] = counts.get(family, 0) + 1
    # A large archive from one build must not drown out independent build families.
    return np.asarray([1.0 / counts[family] for family in family_by_row], dtype=float)


def _risk(candidate: np.ndarray, source_points: np.ndarray,
          source_labels: np.ndarray, source_weights: np.ndarray,
          target_points: np.ndarray, target_labels: np.ndarray,
          config: KernelMemoryConfig) -> tuple[float, float, float]:
    p_source, source_support = _local_evidence(
        candidate, source_points, source_labels, config, source_weights)
    p_target, target_support = _local_evidence(
        candidate, target_points, target_labels, config)
    gate = (config.source_weight * source_support /
            (source_support + config.source_support_scale))
    probability = gate * p_source + (1.0 - gate) * p_target
    uncertainty = np.sqrt(max(probability * (1.0 - probability), 0.0) /
                          (config.prior_strength + source_support + target_support))
    return probability + config.exploration * uncertainty, probability, gate


def run_memory_kde(candidates: list[dict], history: list[dict],
                   oracle: TargetOracle, budget: int, random_seed: int,
                   target_build_id: str, parent_build_id: str | None = None,
                   mode: str = "regression", session_id: str = "memory_kde",
                   config: KernelMemoryConfig | None = None
                   ) -> tuple[list[dict], list[dict]]:
    """Select likely failures from local source/target outcome evidence.

    The estimator uses only normalized scenario inputs, visible historical labels,
    and outcomes already returned by the oracle. Ranks 10 and 20 retain the
    existing global maximin coverage checkpoints.
    """
    if mode not in {"regression", "cross_agent"}:
        raise ValueError(f"unsupported task mode: {mode}")
    if mode == "regression" and any(not isinstance(row.get("parent_pass"), bool)
                                     for row in candidates):
        raise ValueError("regression tasks require explicit parent-pass labels")
    config = config or KernelMemoryConfig()
    history = [row for row in contextual_history(history, candidates)
               if is_usable_outcome(row)]
    source_points = _coordinates(history)
    source_labels = np.asarray([float(row.get("ego_collision") is True)
                                for row in history], dtype=float)
    family_weights = _source_family_weights(history)
    candidate_points = _coordinates(candidates)
    candidate_map = {row["scenario_id"]: row for row in candidates}
    available = set(candidate_map)
    selected: list[dict] = []
    observations: list[dict] = []
    queries: list[dict] = []
    rng = np.random.default_rng(random_seed)

    for rank in range(1, min(int(budget), len(candidates)) + 1):
        pool_indices = [index for index, item in enumerate(candidates)
                        if item["scenario_id"] in available]
        pool = [candidates[index] for index in pool_indices]
        if rank in (10, 20):
            scene = _art_choice(pool, selected, rng)
            reason = "global_art_maximin"
            risk, source_gate = 0.0, 0.0
        else:
            target_points = _coordinates(observations)
            target_labels = np.asarray([float(row.get("ego_collision") is True)
                                        for row in observations], dtype=float)
            scored = [_risk(candidate_points[index], source_points, source_labels,
                            family_weights, target_points, target_labels, config)
                      for index in pool_indices]
            scores = np.asarray([item[0] for item in scored])
            best = float(scores.max())
            ties = np.flatnonzero(np.isclose(scores, best))
            chosen = int(ties[int(rng.integers(len(ties)))])
            scene = pool[chosen]
            risk, _posterior, source_gate = scored[chosen]
            reason = "local_memory_target_evidence"

        outcome = oracle.query(scene["scenario_id"])
        parent_pass = scene.get("parent_pass")
        reward = task_reward(mode, outcome,
                             parent_pass if mode == "regression" else None)
        query = {
            "method": "FBRT-Memory-KDE-v3", "rank": rank,
            "scenario_id": scene["scenario_id"],
            "template_id": scene["template_id"],
            "selection_reason": reason,
            "selected_build_id": target_build_id,
            "session_id": session_id, "mode": mode,
            "ego_collision": outcome.get("ego_collision"),
            "inconclusive": bool(outcome.get("inconclusive", False)),
            "valid_collision": (outcome.get("ego_collision") is True and
                                not outcome.get("inconclusive", False)),
            "regression": (bool(parent_pass and outcome.get("ego_collision") is True and
                                 not outcome.get("inconclusive", False))
                           if mode == "regression" else None),
            "selection_reward": reward,
            "execution_id": outcome.get("execution_id"),
            "episode_cost": int(outcome.get("episode_cost", 0)),
            "predicted_risk": float(risk),
            "source_weight_at_selection": float(source_gate),
        }
        queries.append(query)
        available.remove(scene["scenario_id"])
        selected.append(scene)
        if is_usable_outcome(outcome):
            observed = dict(scene)
            observed.update(outcome)
            observations.append(observed)
    return queries, observations
