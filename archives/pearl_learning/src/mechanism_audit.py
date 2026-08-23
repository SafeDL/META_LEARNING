"""Scripted longitudinal policies and deterministic Gate-1 summaries."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


SCRIPTED_POLICIES = (
    "P0_coast",
    "P1_moderate_accelerate",
    "P2_strong_accelerate",
    "P3_moderate_brake",
    "P4_early_accelerate_then_brake",
    "P5_early_brake_then_accelerate",
    "P6_arrival_gap_heuristic",
    "P7_adversary_first_feedback",
    "P8_sut_first_feedback",
)


def scripted_longitudinal_action(name: str, step: int, observation: np.ndarray, horizon: int, *, action_mode: str = "longitudinal_residual") -> np.ndarray:
    """One-dimensional, route-tracker-compatible policy action."""
    if name not in SCRIPTED_POLICIES:
        raise ValueError(f"unsupported scripted mechanism policy: {name!r}")
    if action_mode == "target_arrival_gap":
        targets = {
            "P0_coast": 0.0, "P1_moderate_accelerate": -0.45,
            "P2_strong_accelerate": -0.90, "P3_moderate_brake": 0.45,
            "P4_early_accelerate_then_brake": -0.70 if step < max(1, int(horizon) // 3) else 0.20,
            "P5_early_brake_then_accelerate": 0.70 if step < max(1, int(horizon) // 3) else -0.55,
            "P6_arrival_gap_heuristic": 0.0,
            "P7_adversary_first_feedback": -0.75,
            "P8_sut_first_feedback": 0.75,
        }
        return np.asarray([targets[name]], dtype=np.float32)
    if action_mode != "longitudinal_residual":
        raise ValueError("unsupported scripted mechanism action mode")
    if name == "P0_coast":
        value = 0.0
    elif name == "P1_moderate_accelerate":
        value = 0.45
    elif name == "P2_strong_accelerate":
        value = 0.90
    elif name == "P3_moderate_brake":
        value = -0.45
    elif name == "P4_early_accelerate_then_brake":
        value = 0.70 if step < max(1, int(horizon) // 3) else -0.20
    elif name == "P5_early_brake_then_accelerate":
        value = -0.70 if step < max(1, int(horizon) // 3) else 0.55
    else:
        # Dynamic observation index 16 is the signed adversary-minus-SUT
        # arrival-time difference, normalized by normalization.time_s.
        gap = float(observation[16])
        if name == "P6_arrival_gap_heuristic":
            value = gap
        elif name == "P7_adversary_first_feedback":
            # Keep the adversary just ahead (negative signed ETA gap), rather
            # than merely driving the pair to exact simultaneity.  The small
            # target is deliberately below the calibrated arrival-gap scale.
            value = 2.0 * (gap + 0.006)
        else:
            # Symmetric SUT-first controller.  P7/P8 are task-conditioned
            # probe candidates only; they are never used as a training label.
            value = 2.0 * (gap - 0.006)
        value = float(np.clip(value, -1.0, 1.0))
    return np.asarray([value], dtype=np.float32)


def rollout_scripted_policy(task: Any, case: Mapping[str, Any], config: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    """Execute exactly one scripted policy under the executable v2 metric."""
    from .task_env import LogicalMergeEnv

    env = LogicalMergeEnv(task, config, [case])
    actions: list[float] = []
    trace: list[dict[str, float]] = []
    try:
        observation, _ = env.reset(options={"case": dict(case)})
        for step in range(int(config["environment"]["horizon"])):
            action = scripted_longitudinal_action(policy_name, step, observation, int(config["environment"]["horizon"]), action_mode=str(config.get("control", {}).get("mechanism_action_mode", "longitudinal_residual")))
            observation, _, terminated, truncated, info = env.step(action)
            actions.append(float(action[0]))
            trace.append({
                "arrival_gap_abs_s": float(info.get("arrival_gap_abs_s", np.nan)),
                "joint_conflict_distance_m": float(info.get("joint_conflict_distance_m", np.nan)),
                "pair_distance_m": float(info.get("pair_distance_m", np.nan)),
                "ttc_s": float(info.get("ttc_s", info.get("ttc", np.nan))),
                "longitudinal_action": float(action[0]),
            })
            if terminated or truncated:
                break
        record = env.episode_record()
    finally:
        env.close()
    return {
        "task_id": task.task_id,
        "case_id": str(case["case_id"]),
        "matched_condition_id": str(case["matched_condition_id"]),
        "policy": policy_name,
        "record": record,
        "longitudinal_actions": actions,
        "mean_longitudinal_action": float(np.mean(actions)) if actions else 0.0,
        "trace": trace,
    }


def _record_score(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    record = row["record"]
    return (
        float(bool(record["valid_critical_strict"])),
        -float(bool(record["target_collision"])),
        -float(bool(record["invalid"])),
        float(record["episode_return"]),
    )


def policy_conflict_report(rows: list[Mapping[str, Any]], task_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create matrix, matched-case rankings, and the predeclared Gate-1 test."""
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("Gate 1 requires exactly two distinct tasks")
    expected = {(task_id, policy) for task_id in task_ids for policy in SCRIPTED_POLICIES}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_id"]), str(row["policy"])), []).append(row)
    if set(grouped) != expected:
        raise ValueError("scripted policy matrix is incomplete")
    matrix: dict[str, Any] = {}
    for task_id in task_ids:
        matrix[task_id] = {}
        for policy in SCRIPTED_POLICIES:
            values = grouped[(task_id, policy)]
            records = [value["record"] for value in values]
            matrix[task_id][policy] = {
                "episodes": len(values),
                "mean_return": float(np.mean([float(record["episode_return"]) for record in records])),
                "valid_critical_strict_rate": float(np.mean([bool(record["valid_critical_strict"]) for record in records])),
                "target_collision_rate": float(np.mean([bool(record["target_collision"]) for record in records])),
                "invalid_rate": float(np.mean([bool(record["invalid"]) for record in records])),
                "median_min_ttc": float(np.median([float(record["min_ttc"]) for record in records])),
                "mean_longitudinal_action": float(np.mean([float(value["mean_longitudinal_action"]) for value in values])),
            }
    aggregate_order = {
        task_id: sorted(
            SCRIPTED_POLICIES,
            key=lambda policy: (
                -matrix[task_id][policy]["valid_critical_strict_rate"],
                matrix[task_id][policy]["target_collision_rate"],
                matrix[task_id][policy]["invalid_rate"],
                -matrix[task_id][policy]["mean_return"],
                policy,
            ),
        )
        for task_id in task_ids
    }
    by_condition: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["matched_condition_id"]), {}).setdefault(str(row["task_id"]), []).append(row)
    rankings: dict[str, Any] = {}
    winner_changes = 0
    for condition_id, task_rows in sorted(by_condition.items()):
        if set(task_rows) != set(task_ids):
            raise ValueError(f"matched condition {condition_id} is absent from one task")
        orders = {
            task_id: [row["policy"] for row in sorted(
                task_rows[task_id], key=lambda row: tuple(-x for x in _record_score(row)) + (str(row["policy"]),)
            )]
            for task_id in task_ids
        }
        winner_changes += int(orders[task_ids[0]][0] != orders[task_ids[1]][0])
        rankings[condition_id] = {"policy_order": orders}
    # A policy must move in opposite directions relative to coast, not merely
    # have different absolute returns because the two geometries differ.
    coast = {task_id: matrix[task_id]["P0_coast"] for task_id in task_ids}
    tradeoff_policies = []
    for policy in SCRIPTED_POLICIES[1:]:
        deltas = [
            matrix[task_id][policy]["valid_critical_strict_rate"] - coast[task_id]["valid_critical_strict_rate"]
            for task_id in task_ids
        ]
        if max(deltas) >= 1.0 / 12.0 and min(deltas) <= -1.0 / 12.0:
            tradeoff_policies.append(policy)
    best_a, best_b = aggregate_order[task_ids[0]][0], aggregate_order[task_ids[1]][0]
    direction_a = float(matrix[task_ids[0]][best_a]["mean_longitudinal_action"])
    direction_b = float(matrix[task_ids[1]][best_b]["mean_longitudinal_action"])
    direction_min_magnitude = 0.10
    criteria = {
        "different_aggregate_best_policy": best_a != best_b,
        "at_least_30pct_matched_case_winner_changes": winner_changes / max(len(rankings), 1) >= 0.30,
        "policy_improves_one_task_and_worsens_other_vs_coast": bool(tradeoff_policies),
        "aggregate_best_longitudinal_directions_are_opposite": bool(
            direction_a * direction_b < 0.0
            and min(abs(direction_a), abs(direction_b)) >= direction_min_magnitude
        ),
    }
    maximum_strict_rate = max(
        matrix[task_id][policy]["valid_critical_strict_rate"]
        for task_id in task_ids for policy in SCRIPTED_POLICIES
    )
    # Ranking policies only by their lack of collision/invalid events is not
    # evidence of an adversarial near-miss decision boundary.  At least one
    # frozen scripted probe must realize the actual Gate-1 search objective.
    objective_observed = bool(maximum_strict_rate > 0.0)
    passed = bool(objective_observed and sum(bool(value) for value in criteria.values()) >= 2)
    gate = {
        "schema": "logical_merge_policy_conflict_gate_v1",
        "gate_name": "task_policy_conflict",
        "status": "pass" if passed else "fail",
        "criteria": criteria,
        "criterion_count": int(sum(bool(value) for value in criteria.values())),
        "objective_evidence": {
            "maximum_valid_critical_strict_rate": float(maximum_strict_rate),
            "strict_objective_observed": objective_observed,
        },
        "aggregate_best_policy": {task_ids[0]: best_a, task_ids[1]: best_b},
        "aggregate_best_mean_longitudinal_action": {task_ids[0]: direction_a, task_ids[1]: direction_b},
        "minimum_mean_action_magnitude_for_directional_conflict": direction_min_magnitude,
        "matched_case_winner_change_rate": winner_changes / max(len(rankings), 1),
        "tradeoff_policies": tradeoff_policies,
        "next_allowed_stage": "gate_1b_single_task_sac" if passed else "revise_task_case_or_reward_before_any_pearl_training",
    }
    return ({"schema": "scripted_policy_matrix_v1", "task_ids": task_ids, "matrix": matrix, "aggregate_policy_order": aggregate_order},
            {"schema": "scripted_policy_case_rankings_v1", "task_ids": task_ids, "cases": rankings}, gate)


def single_task_sac_transfer_report(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]], task_ids: list[str],
) -> dict[str, Any]:
    """Apply the predeclared Gate-1B diagonal-advantage criterion."""
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("Gate 1B requires exactly two distinct tasks")
    if set(matrix) != set(task_ids) or any(set(matrix[source]) != set(task_ids) for source in task_ids):
        raise ValueError("single-task SAC transfer matrix is incomplete")
    first, second = task_ids
    advantages = {
        first: float(matrix[first][first]["valid_critical_strict_rate"])
        - float(matrix[second][first]["valid_critical_strict_rate"]),
        second: float(matrix[second][second]["valid_critical_strict_rate"])
        - float(matrix[first][second]["valid_critical_strict_rate"]),
    }
    episode_counts = {
        target: int(matrix[target][target]["episodes"])
        for target in task_ids
    }
    if any(count <= 0 for count in episode_counts.values()):
        raise ValueError("Gate 1B evaluation requires non-empty matched case sets")
    # A single frozen-case outcome is the smallest non-zero resolution in this
    # deliberately small experiment.  Requiring it on both matrix diagonals
    # prevents a global context-invariant policy from passing the gate.
    minimum_advantages = {target: 1.0 / count for target, count in episode_counts.items()}
    maximum_strict_rate = max(
        float(matrix[source][target]["valid_critical_strict_rate"])
        for source in task_ids for target in task_ids
    )
    passed = bool(
        maximum_strict_rate > 0.0
        and all(advantages[target] >= minimum_advantages[target] for target in task_ids)
    )
    return {
        "schema": "logical_merge_single_task_sac_transfer_gate_v1",
        "gate_name": "single_task_sac_transfer",
        "status": "pass" if passed else "fail",
        "task_ids": task_ids,
        "diagonal_vcsr_advantage": advantages,
        "minimum_advantage_by_task": minimum_advantages,
        "maximum_valid_critical_strict_rate": maximum_strict_rate,
        "next_allowed_stage": "gate_2_context_identifiability" if passed else "revise_task_case_or_reward_before_any_pearl_training",
    }
