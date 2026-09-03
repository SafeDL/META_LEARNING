"""Strict multi-query Cut-in Inner few-shot validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..evaluation.cutin_query_design import (
    CutInValidationQuery,
    build_cutin_validation_queries,
)
from ..evaluation.fewshot_inner import paired_bootstrap, valid_critical_score
from ..evaluation.support_schedule import NestedSupportSchedule
from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
)
from ..training.stages import TrainingStage
from ..training.trainers import build_online


SUPPORT_SHOTS = (0, 1, 2, 4)


def _continuous_risk_score(
    outcome: Mapping[str, Any], criteria: FailureCriteria
) -> float:
    if not bool(outcome.get("is_valid_episode", False)):
        return 0.0
    ttc = max(float(outcome.get("min_ttc", criteria.ttc_s)), 0.0)
    distance = max(float(outcome.get("min_distance", criteria.distance_m)), 0.0)
    closing = max(float(outcome.get("max_closing_speed", 0.0)), 0.0)
    intensity = (
        0.45 * np.exp(-ttc / max(criteria.ttc_s, 1e-6))
        + 0.35 * np.exp(-distance / max(criteria.distance_m, 1e-6))
        + 0.20 * np.clip(
            closing / max(criteria.closing_speed_mps, 1e-6), 0.0, 1.0
        )
    )
    if bool(outcome.get("valid_target_collision", False)):
        intensity = max(float(intensity), 1.0)
    elif bool(outcome.get("valid_critical_near_miss", False)):
        intensity = max(float(intensity), 0.5)
    return float(np.clip(intensity, 0.0, 1.0))


def _post_onset(episode: Any) -> list[Mapping[str, Any]]:
    rows = [
        row for row in episode.rollout.transitions
        if row["info"].get("semantic_maneuver_started", False)
    ]
    return rows or list(episode.rollout.transitions)


def _array(row: Mapping[str, Any], field: str) -> list[float]:
    return [float(value) for value in np.asarray(row[field], dtype=float).tolist()]


def _mean_array(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = np.asarray([row[field] for row in rows], dtype=float)
    return [float(value) for value in values.mean(axis=0).tolist()]


def _record(
    episode: Any,
    task: Any,
    query: CutInValidationQuery,
    shots: int,
    seed: int,
    criteria: FailureCriteria,
    support: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    rows = _post_onset(episode)
    first = rows[0]
    tracking = np.asarray([
        abs(float(row["info"]["maneuver_reference_lateral_error_m"]))
        for row in rows
    ], dtype=float)
    planner_values = np.asarray([row["planner_action"] for row in rows], dtype=float)
    outcome = episode.outcome
    return {
        "task_id": task.task_id,
        "sut_ref": task.sut_ref,
        "geometry_id": task.geometry_id,
        "logical_domain_id": task.logical_domain_id,
        "query_id": query.query_id,
        "query_design_kind": query.design_kind,
        "candidate_index": query.candidate_index,
        "normalized_parameters": [float(value) for value in query.action.continuous],
        "logical_parameters": dict(episode.concrete_scenario.logical_parameters),
        "support_shots": int(shots),
        "seed": int(seed),
        "policy": "shared_prior" if shots == 0 else "adapted_h_z",
        "z": [float(value) for value in episode.latent_before.squeeze(0).tolist()],
        "support_queries": [dict(value) for value in support[:shots]],
        "risk_score": valid_critical_score(outcome),
        "continuous_risk_score": _continuous_risk_score(outcome, criteria),
        "invalid": not bool(outcome.get("is_valid_episode", False)),
        "event_kind": outcome.get("event_kind"),
        "termination_reason": outcome.get("termination_reason"),
        "valid_target_collision": bool(outcome.get("valid_target_collision", False)),
        "valid_critical_near_miss": bool(outcome.get("valid_critical_near_miss", False)),
        "maneuver_completed": any(
            bool(row["info"].get("semantic_maneuver_completed", False))
            for row in episode.rollout.transitions
        ),
        "min_ttc": float(outcome.get("min_ttc", 0.0)),
        "min_distance": float(outcome.get("min_distance", 0.0)),
        "max_closing_speed": float(outcome.get("max_closing_speed", 0.0)),
        "first_planner_action": _array(first, "planner_action"),
        "mean_planner_action": [
            float(value) for value in planner_values.mean(axis=0).tolist()
        ],
        "planner_action_std": [
            float(value) for value in planner_values.std(axis=0).tolist()
        ],
        "first_requested_vehicle_action": _array(
            first, "requested_vehicle_action"
        ),
        "first_executed_vehicle_action": _array(
            first, "executed_vehicle_action"
        ),
        "mean_requested_vehicle_action": _mean_array(
            rows, "requested_vehicle_action"
        ),
        "mean_executed_vehicle_action": _mean_array(
            rows, "executed_vehicle_action"
        ),
        "tracking_rms_m": float(np.sqrt(np.mean(np.square(tracking)))),
        "tracking_p95_m": float(np.quantile(tracking, 0.95)),
        "effective_path_length_mean_m": float(np.mean([
            row["info"]["maneuver_reference_length_m"] for row in rows
        ])),
    }


def _execution_error_record(
    task: Any,
    query: CutInValidationQuery,
    shots: int,
    seed: int,
    latent: torch.Tensor,
    support: Sequence[Mapping[str, object]],
    error: Exception,
) -> dict[str, Any]:
    zeros4, zeros2 = [0.0] * 4, [0.0] * 2
    return {
        "task_id": task.task_id,
        "sut_ref": task.sut_ref,
        "geometry_id": task.geometry_id,
        "logical_domain_id": task.logical_domain_id,
        "query_id": query.query_id,
        "query_design_kind": query.design_kind,
        "candidate_index": query.candidate_index,
        "normalized_parameters": list(query.action.continuous),
        "logical_parameters": {},
        "support_shots": int(shots),
        "seed": int(seed),
        "policy": "shared_prior" if shots == 0 else "adapted_h_z",
        "z": [float(value) for value in latent.squeeze(0).tolist()],
        "support_queries": [dict(value) for value in support[:shots]],
        "risk_score": 0.0,
        "continuous_risk_score": 0.0,
        "invalid": True,
        "execution_error": f"{type(error).__name__}: {error}",
        "event_kind": None,
        "termination_reason": "execution_error",
        "valid_target_collision": False,
        "valid_critical_near_miss": False,
        "maneuver_completed": False,
        "min_ttc": 0.0,
        "min_distance": 0.0,
        "max_closing_speed": 0.0,
        "first_planner_action": zeros4,
        "mean_planner_action": zeros4,
        "planner_action_std": zeros4,
        "first_requested_vehicle_action": zeros2,
        "first_executed_vehicle_action": zeros2,
        "mean_requested_vehicle_action": zeros2,
        "mean_executed_vehicle_action": zeros2,
        "tracking_rms_m": 1_000_000.0,
        "tracking_p95_m": 1_000_000.0,
        "effective_path_length_mean_m": 0.0,
    }


def _paired_k_deltas(
    records: Sequence[Mapping[str, Any]], shots: int, field: str
) -> list[dict[str, float]]:
    indexed = {
        (
            row["task_id"], row["seed"], row["query_id"], row["support_shots"]
        ): row
        for row in records
    }
    pairs = []
    for row in records:
        if int(row["support_shots"]) != 0:
            continue
        key = (row["task_id"], row["seed"], row["query_id"], int(shots))
        adapted = indexed.get(key)
        if adapted is None:
            raise ValueError("incomplete K-shot paired validation records")
        pairs.append({
            "score_delta": float(adapted[field]) - float(row[field]),
            "invalid_delta": float(bool(adapted["invalid"])) - float(bool(row["invalid"])),
        })
    return pairs


def _support_effects(
    records: Sequence[Mapping[str, Any]], support_shots: Sequence[int]
) -> dict[str, Any]:
    indexed = {
        (row["task_id"], row["seed"], row["query_id"], int(row["support_shots"])): row
        for row in records
    }
    report: dict[str, Any] = {}
    for shots in support_shots:
        if shots == 0:
            continue
        z_delta, planner_delta, vehicle_delta = [], [], []
        for key, baseline in indexed.items():
            if key[-1] != 0:
                continue
            adapted = indexed.get(key[:-1] + (int(shots),))
            if adapted is None:
                raise ValueError("incomplete support-effect pairing")
            z_delta.append(float(np.linalg.norm(
                np.asarray(adapted["z"]) - np.asarray(baseline["z"])
            )))
            planner_delta.append(float(np.linalg.norm(
                np.asarray(adapted["mean_planner_action"])
                - np.asarray(baseline["mean_planner_action"])
            )))
            vehicle_delta.append(float(np.linalg.norm(
                np.asarray(adapted["mean_executed_vehicle_action"])
                - np.asarray(baseline["mean_executed_vehicle_action"])
            )))
        report[str(shots)] = {
            "pairs": len(z_delta),
            "mean_z_l2_delta_from_k0": float(np.mean(z_delta)),
            "mean_planner_action_l2_delta_from_k0": float(np.mean(planner_delta)),
            "mean_executed_vehicle_action_l2_delta_from_k0": float(np.mean(vehicle_delta)),
            "z_changed": bool(any(value > 1e-6 for value in z_delta)),
            "planner_action_changed": bool(any(value > 1e-6 for value in planner_delta)),
            "executed_vehicle_action_changed": bool(any(value > 1e-6 for value in vehicle_delta)),
        }
    return report


def _action_variation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if int(row["support_shots"]) == 0]
    actions = np.asarray([row["mean_planner_action"][:3] for row in rows], dtype=float)
    parameters = np.asarray([row["normalized_parameters"] for row in rows], dtype=float)
    standard_deviation = actions.std(axis=0)
    correlations = np.zeros((3, 5), dtype=float)
    for action_index in range(3):
        for parameter_index in range(5):
            if standard_deviation[action_index] > 1e-8 and parameters[:, parameter_index].std() > 1e-8:
                correlations[action_index, parameter_index] = np.corrcoef(
                    actions[:, action_index], parameters[:, parameter_index]
                )[0, 1]
    state_related = np.max(np.abs(correlations), axis=1)
    return {
        "planner_parameter_std": standard_deviation.tolist(),
        "max_abs_query_parameter_correlation": state_related.tolist(),
        "nondegenerate": bool(np.all(standard_deviation > 0.01)),
        "state_related": bool(np.all(state_related > 0.05)),
    }


def _gif_selection(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in records if int(row["support_shots"]) == 0]
    selected = []
    groups = sorted({
        (str(row["task_id"]), int(row["candidate_index"])) for row in rows
    })
    for task_id, candidate in groups:
        candidate_rows = [
            row for row in rows
            if row["task_id"] == task_id
            and int(row["candidate_index"]) == candidate
        ]
        means: dict[str, float] = {}
        for query_id in {str(row["query_id"]) for row in candidate_rows}:
            values = [
                float(row["continuous_risk_score"])
                for row in candidate_rows if row["query_id"] == query_id
            ]
            means[query_id] = float(np.mean(values))
        ordered = sorted(means, key=means.get)
        for label, quantile in (("low", 0.10), ("medium", 0.50), ("high", 0.90)):
            index = int(round(quantile * (len(ordered) - 1)))
            query_id = ordered[index]
            representative = next(
                row for row in candidate_rows if row["query_id"] == query_id
            )
            selected.append({
                "task_id": task_id,
                "candidate_index": candidate,
                "risk_stratum": label,
                "query_id": query_id,
                "mean_k0_continuous_risk": means[query_id],
                "normalized_parameters": representative["normalized_parameters"],
            })
    return selected


def run(config_path: str, checkpoint_path: str) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    cutin_inner = config.get("cutin_inner")
    if cutin_inner is None or bool(cutin_inner.get("allow_outer", True)):
        raise ValueError("validation requires the no-Outer Cut-in configuration")
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    if checkpoint.stage != TrainingStage.CONTEXT_META.value:
        raise ValueError("validation requires a context_meta checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    tasks = select_cutin_validation_tasks(
        load_taskbook(taskbook_path), cutin_inner.get("validation_geometry_ids", ())
    )
    evaluation = config["evaluation"]
    shots_values = tuple(int(value) for value in evaluation["support_shots"])
    if shots_values != SUPPORT_SHOTS:
        raise ValueError("Cut-in validation requires K=0/1/2/4")
    design = evaluation["query_design"]
    candidate_count = len(mvr_parameter_spaces()["cutin"].candidates)
    criteria = FailureCriteria.from_config(config["failure"])
    step_budget = int(config["training"]["step_budget"])
    records: list[dict[str, Any]] = []
    support_provenance: list[dict[str, Any]] = []
    for task in tasks:
        queries = build_cutin_validation_queries(
            task,
            candidates=candidate_count,
            sobol_interior=int(design["sobol_interior_per_candidate"]),
            boundary=int(design["boundary_per_candidate"]),
            seed=int(evaluation["paired_bootstrap_seed"]),
        )
        for seed_value in evaluation["seeds"]:
            seed = int(seed_value)
            online = build_online(model, task, step_budget, criteria)
            support_schedule = NestedSupportSchedule(
                task, max(SUPPORT_SHOTS), seed,
                tuple(query.action for query in queries),
            )
            support_result = online.run(
                task,
                max(SUPPORT_SHOTS),
                deterministic=True,
                posterior_support_limit=max(SUPPORT_SHOTS),
                scene_action_provider=support_schedule,
                episode_seed_provider=lambda current, index, value=seed: int(
                    current.geometry_seed + 10_000 * value + index
                ),
            )
            with torch.no_grad():
                prior, _ = model.context_encoder.prior(device=model.device)
            latents = {
                0: prior.detach().cpu(),
                1: support_result.episodes[0].latent_after,
                2: support_result.episodes[1].latent_after,
                4: support_result.episodes[3].latent_after,
            }
            support_rows = support_schedule.provenance()
            support_provenance.append({
                "task_id": task.task_id,
                "seed": seed,
                "nested_support": support_rows,
            })
            for query_index, query in enumerate(queries):
                episode_seed = int(
                    task.geometry_seed + 100_000 * seed + query_index
                )
                for shots in SUPPORT_SHOTS:
                    try:
                        result = online.run(
                            task,
                            1,
                            deterministic=True,
                            posterior_support_limit=0,
                            initial_latent=latents[shots],
                            scene_action_provider=(
                                lambda _task, _index, _candidates, _space, action=query.action: action
                            ),
                            episode_seed_provider=(
                                lambda _task, _index, value=episode_seed: value
                            ),
                        )
                        records.append(_record(
                            result.episodes[0], task, query, shots, seed,
                            criteria, support_rows,
                        ))
                    except Exception as error:
                        records.append(_execution_error_record(
                            task, query, shots, seed, latents[shots],
                            support_rows, error,
                        ))
    effects = _support_effects(records, SUPPORT_SHOTS)
    samples = int(evaluation["paired_bootstrap_samples"])
    bootstrap_seed = int(evaluation["paired_bootstrap_seed"])
    paired = {
        str(shots): paired_bootstrap(
            _paired_k_deltas(records, shots, "continuous_risk_score"),
            samples=samples,
            seed=bootstrap_seed,
        )
        for shots in SUPPORT_SHOTS[1:]
    }
    valid_rate = float(np.mean([not bool(row["invalid"]) for row in records]))
    non_collision = [
        row for row in records if row["termination_reason"] != "target_collision"
    ]
    completion_rate = float(np.mean([
        bool(row["maneuver_completed"]) for row in non_collision
    ])) if non_collision else 1.0
    tracking_rows = [row for row in records if np.isfinite(row["tracking_rms_m"])]
    tracking_rms = float(np.mean([row["tracking_rms_m"] for row in tracking_rows]))
    tracking_p95 = float(np.quantile([
        row["tracking_p95_m"] for row in tracking_rows
    ], 0.95))
    variation = _action_variation(records)
    expected = (
        len(tasks) * candidate_count
        * (int(design["sobol_interior_per_candidate"]) + int(design["boundary_per_candidate"]))
        * len(evaluation["seeds"]) * len(SUPPORT_SHOTS)
    )
    gates = {
        "all_query_k_pairs_executed": len(records) == expected and not any(
            "execution_error" in row for row in records
        ),
        "valid_episode_rate_at_least_0_95": valid_rate >= 0.95,
        "non_collision_maneuver_completion_at_least_0_90": completion_rate >= 0.90,
        "planner_parameters_nondegenerate": variation["nondegenerate"],
        "planner_parameters_state_related": variation["state_related"],
        "k_positive_changes_z_and_planner_action": all(
            effects[str(shots)]["z_changed"]
            and effects[str(shots)]["planner_action_changed"]
            for shots in SUPPORT_SHOTS[1:]
        ),
        "k4_continuous_risk_ci_nonnegative": paired["4"]["ci95_lower"] >= 0.0,
    }
    return {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "validation",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint_stage": checkpoint.stage,
        "query_protocol": {
            "continuous_dimensions": 5,
            "candidate_is_discrete_logical_parameter": True,
            "sobol_interior_per_candidate": int(design["sobol_interior_per_candidate"]),
            "boundary_per_candidate": int(design["boundary_per_candidate"]),
            "queries_per_candidate": int(design["sobol_interior_per_candidate"]) + int(design["boundary_per_candidate"]),
            "query_repetitions": list(evaluation["seeds"]),
            "support_sets": "task_seed_local_nested_K_1_2_4_disjoint_from_all_queries",
            "adaptation": "posterior_z_inference_only_policy_weights_frozen",
        },
        "support_provenance": support_provenance,
        "query_records": records,
        "support_effects": effects,
        "paired_continuous_risk_improvement_k_minus_k0": paired,
        "planner_action_variation": variation,
        "tracking_summary": {
            "mean_episode_rms_m": tracking_rms,
            "episode_p95_tracking_p95_m": tracking_p95,
        },
        "valid_episode_rate": valid_rate,
        "non_collision_maneuver_completion_rate": completion_rate,
        "expected_query_k_episodes": expected,
        "recorded_query_k_episodes": len(records),
        "gif_selection": _gif_selection(records),
        "stage1_gates": gates,
        "stage1_passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
