"""Cheap, deterministic policy calibration for the method-flow benchmark."""
from __future__ import annotations

from typing import Any, Callable, Mapping
import itertools
import numpy as np

from .critical import (
    CRITICAL_METRIC_SCHEMA,
    LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA,
)
from .io import content_hash


CALIBRATION_SCHEMA = "merge_benchmark_calibration_v1"
POLICIES = ("zero", "random", "heuristic")


def longitudinal_policy(
    name: str,
    *,
    case_id: str,
    initial_arrival_gap_s: float,
    observation_time_scale_s: float = 10.0,
    initial_gap_normalizer_s: float | None = None,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Return one of the three frozen calibration policies."""
    if name not in POLICIES:
        raise ValueError(f"unsupported calibration policy: {name!r}")
    rng = np.random.default_rng(int(content_hash({"case_id": case_id, "policy": name})[:16], 16))
    random_action = 0.0

    def policy(step: int, observation: np.ndarray) -> np.ndarray:
        nonlocal random_action
        if name == "zero":
            longitudinal = 0.0
        elif name == "random":
            if step % 10 == 0:
                random_action = float(rng.uniform(-1.0, 1.0))
            longitudinal = random_action
        else:
            # Dynamic observation index 16 is (adversary TTC - SUT TTC),
            # normalized by ``normalization.time_s``. Normalize the live error
            # by this case's initial error: the policy starts with meaningful
            # authority and tapers continuously as the arrival times align.
            current_normalized_gap = float(observation[16])
            current_gap_s = current_normalized_gap * float(observation_time_scale_s)
            normalizer = max(abs(float(initial_arrival_gap_s)), 0.1)
            magnitude = float(np.clip(abs(current_gap_s) / normalizer, 0.0, 1.0))
            longitudinal = float(np.sign(current_gap_s) * magnitude)
        return np.asarray([0.0, longitudinal], dtype=np.float32)

    return policy


def _episode_hit(trace: list[Mapping[str, Any]], thresholds: Mapping[str, float]) -> bool:
    return any(
        float(row["arrival_gap_abs_s"]) <= float(thresholds["arrival_gap_threshold_s"])
        and float(row["joint_conflict_distance_m"]) <= float(thresholds["joint_conflict_distance_threshold_m"])
        and float(row["pair_distance_m"]) <= float(thresholds["pair_distance_threshold_m"])
        and not bool(row.get("physical_target_contact", False))
        for row in trace
    )


episode_near_miss = _episode_hit


def run_baseline_rollout(task: Any, case: dict[str, Any], config: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    """Execute one frozen calibration policy without enabling strict termination."""
    from .task_env import LogicalMergeEnv

    cfg = dict(config)
    trace_cfg = {
        **cfg,
        "critical_metric": {**dict(cfg["critical_metric"]), "calibration_trace_mode": True},
        "reward": {**dict(cfg["reward"]), "valid_critical_bonus": 0.0},
    }
    env = LogicalMergeEnv(task, trace_cfg, [case])
    trace: list[dict[str, Any]] = []
    initial_gap = float("nan")
    final_info: dict[str, Any] = {}
    try:
        observation, _ = env.reset(options={"case": case})
        state = env.initial_case_measurements()
        initial_gap = float(state["adversary_time_s"]) - float(state["sut_time_s"])
        policy = longitudinal_policy(
            policy_name,
            case_id=str(case["case_id"]),
            initial_arrival_gap_s=initial_gap,
            observation_time_scale_s=float(cfg["normalization"]["time_s"]),
            initial_gap_normalizer_s=float(cfg["case_sampling"]["max_initial_arrival_gap_s"]),
        )
        for step in range(int(cfg["environment"]["horizon"])):
            observation, _, terminated, truncated, info = env.step(policy(step, observation))
            final_info = dict(info)
            trace.append({
                key: info[key]
                for key in (
                    "arrival_gap_abs_s", "joint_conflict_distance_m", "pair_distance_m",
                    "ttc_s", "closing_speed_mps", "critical_margin", "physical_target_contact",
                )
            })
            if terminated or truncated:
                break
    finally:
        env.close()
    invalid = any(bool(final_info.get(key, False)) for key in (
        "non_target_collision", "adversary_out_of_road", "sut_out_of_road", "wrong_route"
    ))
    return {
        "task_id": task.task_id,
        "logical_type": task.logical_type,
        "case_id": str(case["case_id"]),
        "policy": policy_name,
        "initial_arrival_gap_s": initial_gap,
        "collision": bool(final_info.get("physical_target_contact", False)),
        "invalid": invalid,
        "episode_steps": len(trace),
        "termination_reason": final_info.get("termination_reason"),
        "terminal_flags": {
            key: bool(final_info.get(key, False))
            for key in (
                "target_collision", "non_target_collision", "adversary_out_of_road",
                "sut_out_of_road", "wrong_route", "adversary_route_complete", "sut_route_complete",
            )
        },
        "terminal_lane_indexes": {
            "adversary": final_info.get("adversary_lane_index"),
            "sut": final_info.get("sut_lane_index"),
        },
        "case": dict(case),
        "trace": trace,
    }


def _rates(rollouts: list[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        rows = [row for row in rollouts if row["policy"] == policy]
        result[policy] = {
            "near_miss_rate": float(np.mean([_episode_hit(row["trace"], thresholds) for row in rows])),
            "collision_rate": float(np.mean([bool(row["collision"]) for row in rows])),
            "invalid_rate": float(np.mean([bool(row["invalid"]) for row in rows])),
        }
    return result


def calibrate_thresholds(
    rollouts: list[Mapping[str, Any]],
    *,
    min_arrival_gap_threshold_s: float = 0.0,
) -> dict[str, Any]:
    """Select the strictest empirical threshold triple satisfying frozen gates."""
    if not rollouts or {str(row["policy"]) for row in rollouts} != set(POLICIES):
        raise ValueError("calibration requires non-empty zero/random/heuristic rollouts")
    task_ids = sorted({str(row.get("task_id", "__aggregate__")) for row in rollouts})
    if len(task_ids) > 1:
        epsilon = {
            "arrival_gap_threshold_s": np.finfo(float).eps,
            "joint_conflict_distance_threshold_m": np.finfo(float).eps,
            "pair_distance_threshold_m": np.finfo(float).eps,
        }
        aggregate_rates = _rates(rollouts, epsilon)
        if (
            aggregate_rates["zero"]["collision_rate"] > 0.10
            or aggregate_rates["random"]["collision_rate"] > 0.10
            or aggregate_rates["heuristic"]["invalid_rate"] > 0.10
        ):
            return {
                "schema": CALIBRATION_SCHEMA,
                "critical_metric_schema": CRITICAL_METRIC_SCHEMA,
                "status": "fail",
                "reason": "aggregate collision/invalid gates fail before task-profile selection",
                "rollout_count": len(rollouts),
            }
        profiles: dict[str, dict[str, float]] = {}
        profile_rates: dict[str, Any] = {}
        for task_id in task_ids:
            subset = [row for row in rollouts if str(row.get("task_id")) == task_id]
            child = calibrate_thresholds(
                subset,
                min_arrival_gap_threshold_s=min_arrival_gap_threshold_s,
            )
            if child["status"] != "pass":
                return {
                    "schema": CALIBRATION_SCHEMA,
                    "critical_metric_schema": CRITICAL_METRIC_SCHEMA,
                    "status": "fail",
                    "reason": f"no feasible validation threshold profile for {task_id}",
                    "rollout_count": len(rollouts),
                }
            logical_type = str(subset[0].get("logical_type", ""))
            if not logical_type:
                logical_type = "bottleneck_merge" if "bottleneck" in task_id else "lane_drop_merge"
            profiles[logical_type] = dict(child["thresholds"])
            profile_rates[logical_type] = dict(child["rates"])
        keys = ("arrival_gap_threshold_s", "joint_conflict_distance_threshold_m", "pair_distance_threshold_m")
        default = {key: min(profile[key] for profile in profiles.values()) for key in keys}
        payload = {
            "schema": CALIBRATION_SCHEMA,
            "critical_metric_schema": CRITICAL_METRIC_SCHEMA,
            "status": "pass",
            "thresholds": default,
            "threshold_profiles": profiles,
            "profile_rates": profile_rates,
            "rates": aggregate_rates,
            "rollout_count": len(rollouts),
            "uses_splits": ["meta_validation"],
            "uses_test_or_ood": False,
            "ood_profile_policy": "componentwise_strictest_validation_profile",
            "policy_contract": {
                "zero": "[steering_residual=0,longitudinal=0]",
                "random": "steering_residual=0; seeded longitudinal U(-1,1) held for 10 steps",
                "heuristic": "current arrival-gap direction and magnitude normalized by the case initial absolute gap",
            },
            "resolution_contract": {
                "min_arrival_gap_threshold_s": float(min_arrival_gap_threshold_s),
                "rationale": "not below one configured environment decision step",
                "empirical_quantile_grid": {"start": 0.025, "stop": 0.975, "points": 37},
            },
        }
        return {**payload, "calibration_hash": content_hash(payload)}
    episode_minima = {
        key: np.asarray([min(float(step[key]) for step in row["trace"]) for row in rollouts], dtype=float)
        for key in ("arrival_gap_abs_s", "joint_conflict_distance_m", "pair_distance_m")
    }
    # A coarse decile grid can skip a narrow heuristic-response mode and jump
    # directly from 10% to 100% episode hits. This remains a deterministic
    # empirical-quantile search, with enough resolution for a 20%-60% gate.
    quantiles = np.linspace(0.025, 0.975, 37)
    floors = {
        "arrival_gap_abs_s": max(float(min_arrival_gap_threshold_s), np.finfo(float).eps),
        "joint_conflict_distance_m": np.finfo(float).eps,
        "pair_distance_m": np.finfo(float).eps,
    }
    grids = {
        key: np.asarray([
            value for value in np.unique(np.quantile(values, quantiles))
            if np.isfinite(value) and value >= floors[key]
        ], dtype=float)
        for key, values in episode_minima.items()
    }
    if any(len(values) == 0 for values in grids.values()):
        return {
            "schema": CALIBRATION_SCHEMA,
            "critical_metric_schema": CRITICAL_METRIC_SCHEMA,
            "status": "fail",
            "reason": "empirical threshold grid contains no finite positive candidate",
            "rollout_count": len(rollouts),
        }
    feasible = []
    for gap, joint, pair in itertools.product(
        grids["arrival_gap_abs_s"], grids["joint_conflict_distance_m"], grids["pair_distance_m"]
    ):
        thresholds = {
            "arrival_gap_threshold_s": float(gap),
            "joint_conflict_distance_threshold_m": float(joint),
            "pair_distance_threshold_m": float(pair),
        }
        rates = _rates(rollouts, thresholds)
        local_task_ids = sorted({str(row.get("task_id", "__aggregate__")) for row in rollouts})
        task_heuristic_rates = {
            task_id: _rates(
                [row for row in rollouts if str(row.get("task_id", "__aggregate__")) == task_id],
                thresholds,
            )["heuristic"]
            for task_id in local_task_ids
        }
        if (
            rates["zero"]["near_miss_rate"] <= 0.10
            and rates["random"]["near_miss_rate"] <= 0.10
            and rates["zero"]["collision_rate"] <= 0.10
            and rates["random"]["collision_rate"] <= 0.10
            and 0.20 <= rates["heuristic"]["near_miss_rate"] <= 0.60
            and rates["heuristic"]["invalid_rate"] <= 0.10
            and all(
                0.20 <= task_rates["near_miss_rate"] <= 0.60
                and task_rates["invalid_rate"] <= 0.10
                for task_rates in task_heuristic_rates.values()
            )
        ):
            normalized_volume = (
                float(np.searchsorted(grids["arrival_gap_abs_s"], gap) + 1)
                * float(np.searchsorted(grids["joint_conflict_distance_m"], joint) + 1)
                * float(np.searchsorted(grids["pair_distance_m"], pair) + 1)
            )
            separation = rates["heuristic"]["near_miss_rate"] - max(
                rates["zero"]["near_miss_rate"], rates["random"]["near_miss_rate"]
            )
            feasible.append((normalized_volume, -separation, thresholds, rates, task_heuristic_rates))
    if not feasible:
        return {
            "schema": CALIBRATION_SCHEMA,
            "critical_metric_schema": CRITICAL_METRIC_SCHEMA,
            "status": "fail",
            "reason": "no empirical threshold triple satisfies the frozen policy-rate gates",
            "rollout_count": len(rollouts),
        }
    _, _, thresholds, rates, task_heuristic_rates = min(
        feasible, key=lambda row: (row[0], row[1], tuple(row[2].values()))
    )
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "critical_metric_schema": CRITICAL_METRIC_SCHEMA,
        "status": "pass",
        "thresholds": thresholds,
        "rates": rates,
        "heuristic_rates_by_validation_task": task_heuristic_rates,
        "rollout_count": len(rollouts),
        "uses_splits": ["meta_validation"],
        "uses_test_or_ood": False,
        "policy_contract": {
            "zero": "[steering_residual=0,longitudinal=0]",
            "random": "steering_residual=0; seeded longitudinal U(-1,1) held for 10 steps",
            "heuristic": "current arrival-gap direction and magnitude normalized by the case initial absolute gap",
        },
        "resolution_contract": {
            "min_arrival_gap_threshold_s": float(min_arrival_gap_threshold_s),
            "rationale": "not below one configured environment decision step",
            "empirical_quantile_grid": {"start": 0.025, "stop": 0.975, "points": 37},
        },
    }
    return {**payload, "calibration_hash": content_hash(payload)}


def apply_calibration_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a passed validation-only calibration into a run configuration."""
    if manifest.get("schema") != CALIBRATION_SCHEMA or manifest.get("status") != "pass":
        raise ValueError("a passed merge benchmark calibration manifest is required")
    if manifest.get("critical_metric_schema") != CRITICAL_METRIC_SCHEMA:
        raise ValueError("calibration critical metric schema is incompatible")
    if bool(manifest.get("uses_test_or_ood", True)):
        raise ValueError("threshold calibration must not use test or OOD data")
    expected_hash = content_hash({key: value for key, value in manifest.items() if key != "calibration_hash"})
    if manifest.get("calibration_hash") != expected_hash:
        raise ValueError("calibration manifest hash mismatch")
    result = dict(config)
    requested_schema = str(config.get("critical_metric", {}).get("schema", CRITICAL_METRIC_SCHEMA))
    if requested_schema not in {CRITICAL_METRIC_SCHEMA, LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA}:
        raise ValueError("configuration requests an unsupported calibrated critical metric schema")
    result["critical_metric"] = {
        **dict(config.get("critical_metric", {})),
        **dict(manifest["thresholds"]),
        "schema": requested_schema,
        "calibration_hash": expected_hash,
    }
    if requested_schema != CRITICAL_METRIC_SCHEMA:
        result["critical_metric"]["threshold_source_metric_schema"] = CRITICAL_METRIC_SCHEMA
    if "threshold_profiles" in manifest:
        result["critical_metric"]["threshold_profiles"] = {
            str(key): dict(value) for key, value in manifest["threshold_profiles"].items()
        }
    return result


def thresholds_for_task(config: Mapping[str, Any], task: Any) -> dict[str, Any]:
    metric = dict(config["critical_metric"])
    profile = dict(metric.get("threshold_profiles", {}).get(str(task.logical_type), {}))
    return {**metric, **profile}
