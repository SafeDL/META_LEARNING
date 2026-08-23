"""Leakage-free trajectory summaries and minimal context-identifiability probe."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .mechanism_audit import scripted_longitudinal_action


def collect_probe_trajectory(task: Any, case: Mapping[str, Any], config: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    """Collect a support trajectory from transitions only (no task descriptor)."""
    from .task_env import LogicalMergeEnv

    env = LogicalMergeEnv(task, config, [case])
    transitions = []
    try:
        observation, _ = env.reset(options={"case": dict(case)})
        for step in range(int(config["environment"]["horizon"])):
            action = scripted_longitudinal_action(policy_name, step, observation, int(config["environment"]["horizon"]))
            next_observation, reward, terminated, truncated, _ = env.step(action)
            transitions.append({
                "observation": observation.astype(np.float32),
                "action": action.astype(np.float32),
                "reward": float(reward),
                "next_observation": next_observation.astype(np.float32),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    if not transitions:
        raise RuntimeError("probe policy produced an empty trajectory")
    return {"case_id": str(case["case_id"]), "transitions": transitions}


def transition_only_feature(trajectory: Mapping[str, Any]) -> np.ndarray:
    """Fixed summary of (obs, action, reward, next_obs, termination) only."""
    rows = list(trajectory["transitions"])
    observation = np.stack([row["observation"] for row in rows])
    action = np.stack([row["action"] for row in rows])
    reward = np.asarray([row["reward"] for row in rows], dtype=np.float32)[:, None]
    next_observation = np.stack([row["next_observation"] for row in rows])
    values = np.concatenate([observation, action, reward, next_observation], axis=1)
    summary = np.concatenate([
        values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0),
        np.asarray([
            len(rows), float(rows[-1]["terminated"]), float(rows[-1]["truncated"]),
        ], dtype=np.float32),
    ])
    if not np.all(np.isfinite(summary)):
        raise ValueError("transition-only feature contains a non-finite value")
    return summary.astype(np.float64)


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, iterations: int = 1_500, l2: float = 0.05) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-6, x.std(axis=0), 1.0)
    normalized = (x - mean) / scale
    weights = np.zeros(normalized.shape[1], dtype=float)
    bias = 0.0
    # Fixed full-batch optimization is deterministic and deliberately tiny;
    # this is an information probe, not a learned task representation.
    for _ in range(iterations):
        logits = np.clip(normalized @ weights + bias, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        weights -= 0.10 * ((normalized.T @ error) / len(y) + l2 * weights)
        bias -= 0.10 * float(error.mean())
    return weights, bias, mean, scale


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    def average_pairwise(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).mean())
    return 2.0 * average_pairwise(left, right) - average_pairwise(left, left) - average_pairwise(right, right)


def context_identifiability_report(
    task_ids: list[str],
    trajectories: Mapping[str, list[Mapping[str, Any]]],
    *,
    bootstrap_samples: int = 400,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return held-out logistic accuracy and an energy-distance separation test."""
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("context identifiability gate requires exactly two tasks")
    features = {task_id: np.stack([transition_only_feature(row) for row in trajectories[task_id]]) for task_id in task_ids}
    if min(len(features[task_id]) for task_id in task_ids) < 6:
        raise ValueError("each task requires at least six support trajectories")
    # Deterministic 2/3 train and 1/3 held-out split by matched-case order.
    train_indexes = {task_id: np.asarray([index for index in range(len(features[task_id])) if index % 3 != 2]) for task_id in task_ids}
    eval_indexes = {task_id: np.asarray([index for index in range(len(features[task_id])) if index % 3 == 2]) for task_id in task_ids}
    train_x = np.concatenate([features[task_id][train_indexes[task_id]] for task_id in task_ids])
    train_y = np.concatenate([np.full(len(train_indexes[task_id]), label, dtype=float) for label, task_id in enumerate(task_ids)])
    test_x = np.concatenate([features[task_id][eval_indexes[task_id]] for task_id in task_ids])
    test_y = np.concatenate([np.full(len(eval_indexes[task_id]), label, dtype=int) for label, task_id in enumerate(task_ids)])
    weights, bias, mean, scale = _fit_logistic(train_x, train_y)
    probs = 1.0 / (1.0 + np.exp(-np.clip(((test_x - mean) / scale) @ weights + bias, -30.0, 30.0)))
    predictions = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == test_y))
    # Feature standardization is fitted on train trajectories only, avoiding
    # held-out label leakage while making the distance interpretable.
    standardized = {task_id: (features[task_id] - mean) / scale for task_id in task_ids}
    energy = _energy_distance(standardized[task_ids[0]], standardized[task_ids[1]])
    rng = np.random.default_rng(20260817)
    samples = []
    for _ in range(int(bootstrap_samples)):
        left = standardized[task_ids[0]][rng.integers(len(standardized[task_ids[0]]), size=len(standardized[task_ids[0]]))]
        right = standardized[task_ids[1]][rng.integers(len(standardized[task_ids[1]]), size=len(standardized[task_ids[1]]))]
        samples.append(_energy_distance(left, right))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    metrics = {
        "schema": "context_probe_metrics_v1",
        "task_ids": task_ids,
        "probe": "l2_regularized_logistic_regression_on_transition_only_summary",
        "feature_contract": {
            "included": ["observation", "action", "reward", "next_observation", "termination"],
            "excluded": ["task_id", "geometry_id", "descriptor", "case_id"],
            "observation_schema": "logical_merge_dynamic_obs_v1",
        },
        "train_trajectories": int(len(train_x)),
        "held_out_trajectories": int(len(test_x)),
        "held_out_accuracy": accuracy,
        "held_out_predictions": predictions.tolist(),
        "held_out_labels": test_y.tolist(),
    }
    distance = {
        "schema": "context_feature_distance_v1",
        "metric": "energy_distance_on_train_standardized_transition_summaries",
        "energy_distance": float(energy),
        "bootstrap_ci_95": [float(lower), float(upper)],
        "bootstrap_samples": int(bootstrap_samples),
    }
    separated = bool(lower > 0.0)
    passed = bool(accuracy >= 0.80 and separated)
    gate = {
        "schema": "logical_merge_context_identifiability_gate_v1",
        "gate_name": "context_identifiability",
        "status": "pass" if passed else "fail",
        "criteria": {"held_out_accuracy_at_least_0_80": accuracy >= 0.80, "stable_energy_distance": separated},
        "next_allowed_stage": "gate_3_vanilla_pearl" if passed else "revise_support_protocol_before_any_pearl_training",
    }
    return metrics, distance, gate
