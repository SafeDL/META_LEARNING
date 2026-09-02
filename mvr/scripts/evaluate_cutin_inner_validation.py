"""Validate Cut-in Inner adaptation without using a test split or Outer policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluation.fewshot_inner import paired_bootstrap, paired_policy_deltas, valid_critical_score
from ..evaluation.support_schedule import FixedQuerySupportSchedule
from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.parameter_space import NormalizedScenarioAction
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


def _fixed_x0(config: Mapping[str, Any]) -> NormalizedScenarioAction:
    payload = config["evaluation"]["fixed_query_x0"]
    return NormalizedScenarioAction(
        int(payload["candidate_index"]),
        tuple(float(value) for value in payload["continuous"]),
    )


def _episode_seed(task: Any, index: int, shots: int, max_support: int, seed: int) -> int:
    source = index if index < shots else max_support + index - shots
    return int(task.geometry_seed + 100_000 * int(seed) + source)


def _first_post_onset_transition(episode: Any) -> Mapping[str, Any]:
    """Return the first transition after semantic onset.

    The raw action is the SAC output in its declared actuator space.  The
    executed action is the physically projected command and can legitimately
    be identical across contexts when the safety envelope is active.  Keep
    both values in the report so adaptation is measured on the policy output,
    not only on the downstream shield.
    """
    transition = episode.rollout.transitions[0]
    for candidate in episode.rollout.transitions:
        if candidate["info"].get("semantic_maneuver_started", False):
            transition = candidate
            break
    return transition


def _action_from_transition(transition: Mapping[str, Any], field: str) -> list[float]:
    value = transition.get(field, transition["action"])
    return [float(item) for item in np.asarray(value, dtype=float).tolist()]


def _continuous_risk_score(outcome: Mapping[str, Any], criteria: FailureCriteria) -> float:
    """Measure event intensity without altering formal failure semantics."""
    if not bool(outcome.get("is_valid_episode", False)):
        return 0.0
    ttc = max(float(outcome.get("min_ttc", criteria.ttc_s)), 0.0)
    distance = max(float(outcome.get("min_distance", criteria.distance_m)), 0.0)
    closing = max(float(outcome.get("max_closing_speed", 0.0)), 0.0)
    ttc_term = float(np.exp(-ttc / max(criteria.ttc_s, 1e-6)))
    distance_term = float(np.exp(-distance / max(criteria.distance_m, 1e-6)))
    closing_term = float(np.clip(closing / max(criteria.closing_speed_mps, 1e-6), 0.0, 1.0))
    intensity = 0.45 * ttc_term + 0.35 * distance_term + 0.20 * closing_term
    if bool(outcome.get("valid_target_collision", False)):
        intensity = max(intensity, 1.0)
    elif bool(outcome.get("valid_critical_near_miss", False)):
        intensity = max(intensity, 0.5)
    return float(np.clip(intensity, 0.0, 1.0))


def _mean_post_onset_action(episode: Any, field: str) -> list[float]:
    transitions = [
        transition for transition in episode.rollout.transitions
        if transition["info"].get("semantic_maneuver_started", False)
    ]
    if not transitions:
        transitions = list(episode.rollout.transitions)
    values = np.asarray([
        _action_from_transition(transition, field) for transition in transitions
    ], dtype=float)
    return [float(item) for item in values.mean(axis=0).tolist()]


def _records(
    episodes: Sequence[Any], task: Any, shots: int, seed: int, policy: str,
    x0: NormalizedScenarioAction, criteria: FailureCriteria,
    support_queries: Sequence[Mapping[str, object]], support_used: bool,
) -> list[dict[str, Any]]:
    def record(episode: Any, query_case_id: int) -> dict[str, Any]:
        transition = _first_post_onset_transition(episode)
        return {
            "task_id": task.task_id,
            "sut_ref": task.sut_ref,
            "functional_scenario": task.functional_scenario,
            "geometry_id": task.geometry_id,
            "logical_domain_id": task.logical_domain_id,
            "support_shots": int(shots),
            "seed": int(seed),
            "query_case_id": query_case_id,
            "policy": policy,
            "risk_score": valid_critical_score(episode.outcome),
            "score": valid_critical_score(episode.outcome),
            "continuous_risk_score": _continuous_risk_score(episode.outcome, criteria),
            "min_ttc": float(episode.outcome.get("min_ttc", 0.0)),
            "min_distance": float(episode.outcome.get("min_distance", 0.0)),
            "max_closing_speed": float(episode.outcome.get("max_closing_speed", 0.0)),
            "invalid": not bool(episode.outcome.get("is_valid_episode", False)),
            "event_kind": episode.outcome.get("event_kind"),
            "termination_reason": episode.outcome.get("termination_reason"),
            "valid_target_collision": bool(episode.outcome.get("valid_target_collision", False)),
            "valid_critical_near_miss": bool(episode.outcome.get("valid_critical_near_miss", False)),
            "traffic_violation_counts": dict(
                episode.outcome.get("traffic_telemetry", {}).get("violation_counts", {})
            ),
            "traffic_warning_counts": dict(
                episode.outcome.get("traffic_telemetry", {}).get("warning_counts", {})
            ),
            "z": [float(value) for value in episode.latent_before.squeeze(0).tolist()],
            # ``first_inner_action`` is deliberately the raw SAC output.  A
            # separate executed field exposes any shield projection.
            "first_inner_action": _action_from_transition(transition, "raw_action"),
            "first_executed_action": _action_from_transition(transition, "executed_action"),
            "mean_post_onset_inner_action": _mean_post_onset_action(episode, "raw_action"),
            "mean_post_onset_executed_action": _mean_post_onset_action(episode, "executed_action"),
            "fixed_query_x0": {
                "candidate_index": x0.candidate_index,
                "continuous": list(x0.continuous),
            },
            "support_queries": [dict(action) for action in support_queries],
            "support_used": bool(support_used),
        }
    return [
        record(episode, query_case_id)
        for query_case_id, episode in enumerate(episodes)
    ]


def _evaluate_task(
    model: Any, task: Any, criteria: FailureCriteria, shots: int, queries: int,
    max_support: int, seed: int, x0: NormalizedScenarioAction, step_budget: int,
    shared_prior: Sequence[Any],
) -> list[dict[str, Any]]:
    provider = FixedQuerySupportSchedule(task, x0, shots, max_support, seed)
    if shots == 0:
        adapted = shared_prior
    else:
        online = build_online(model, task, step_budget, criteria)
        adapted_result = online.run(
            task, shots + queries, deterministic=True, posterior_support_limit=shots,
            scene_action_provider=provider,
            episode_seed_provider=lambda current, index: _episode_seed(
                current, index, shots, max_support, seed,
            ),
        )
        adapted = adapted_result.episodes[shots:]
    support_queries = provider.provenance()
    return (
        _records(
            adapted, task, shots, seed, "adapted_h_z", x0, criteria,
            support_queries, True,
        )
        + _records(
            shared_prior, task, shots, seed, "shared_prior", x0, criteria,
            support_queries, False,
        )
    )


def _shared_prior_episodes(
    model: Any, task: Any, criteria: FailureCriteria, queries: int,
    max_support: int, seed: int, x0: NormalizedScenarioAction, step_budget: int,
) -> Sequence[Any]:
    online = build_online(model, task, step_budget, criteria)
    return online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=lambda _task, _index, _candidates, _space: x0,
        episode_seed_provider=lambda current, index: _episode_seed(
            current, index, 0, max_support, seed,
        ),
    ).episodes


def _paired_field_deltas(
    records: Sequence[Mapping[str, Any]], left_policy: str, right_policy: str,
    field: str,
) -> list[dict[str, float]]:
    key_fields = ("task_id", "logical_domain_id", "support_shots", "seed", "query_case_id")
    indexed: dict[tuple[object, ...], Mapping[str, Any]] = {}
    for record in records:
        if record.get("policy") not in {left_policy, right_policy}:
            continue
        key = tuple(record[field_name] for field_name in key_fields) + (record["policy"],)
        indexed[key] = record
    pairs = []
    for key in sorted({key[:-1] for key in indexed}):
        left = indexed.get(key + (left_policy,))
        right = indexed.get(key + (right_policy,))
        if left is None or right is None:
            raise ValueError("incomplete paired risk evaluation record")
        pairs.append({
            "score_delta": float(left[field]) - float(right[field]),
            "invalid_delta": float(bool(left["invalid"])) - float(bool(right["invalid"])),
        })
    return pairs


def _support_effects(records: Sequence[Mapping[str, Any]], support_shots: Sequence[int]) -> dict[str, Any]:
    adapted = [row for row in records if row["policy"] == "adapted_h_z"]
    key_fields = ("task_id", "seed", "query_case_id")
    indexed = {
        (int(row["support_shots"]),) + tuple(row[field] for field in key_fields): row
        for row in adapted
    }
    baseline = {
        key[1:]: row for key, row in indexed.items() if key[0] == 0
    }
    report: dict[str, Any] = {}
    for shots in support_shots:
        if shots == 0:
            continue
        deltas = []
        for key, baseline_row in baseline.items():
            row = indexed.get((int(shots),) + key)
            if row is None:
                raise ValueError("incomplete fixed-query support comparison")
            deltas.append((
                float(np.linalg.norm(np.asarray(row["z"]) - np.asarray(baseline_row["z"]))),
                float(np.linalg.norm(
                    np.asarray(row["first_inner_action"])
                    - np.asarray(baseline_row["first_inner_action"])
                )),
                float(np.linalg.norm(
                    np.asarray(row.get("first_executed_action", row["first_inner_action"]))
                    - np.asarray(baseline_row.get("first_executed_action", baseline_row["first_inner_action"]))
                )),
                float(np.linalg.norm(
                    np.asarray(row.get("mean_post_onset_inner_action", row["first_inner_action"]))
                    - np.asarray(baseline_row.get("mean_post_onset_inner_action", baseline_row["first_inner_action"]))
                )),
                float(np.linalg.norm(
                    np.asarray(row.get("mean_post_onset_executed_action", row.get("first_executed_action", row["first_inner_action"])))
                    - np.asarray(baseline_row.get("mean_post_onset_executed_action", baseline_row.get("first_executed_action", baseline_row["first_inner_action"])))
                )),
            ))
        z_delta = [row[0] for row in deltas]
        action_delta = [row[1] for row in deltas]
        executed_delta = [row[2] for row in deltas]
        mean_action_delta = [row[3] for row in deltas]
        mean_executed_delta = [row[4] for row in deltas]
        report[str(shots)] = {
            "pairs": len(deltas),
            "mean_z_l2_delta_from_k0": float(np.mean(z_delta)),
            "mean_first_inner_action_l2_delta_from_k0": float(np.mean(action_delta)),
            "z_changed": bool(any(value > 1e-6 for value in z_delta)),
            "inner_action_changed": bool(any(value > 1e-6 for value in action_delta)),
            "mean_first_executed_action_l2_delta_from_k0": float(np.mean(executed_delta)),
            "executed_action_changed": bool(any(value > 1e-6 for value in executed_delta)),
            "mean_post_onset_inner_action_l2_delta_from_k0": float(np.mean(mean_action_delta)),
            "mean_post_onset_inner_action_changed": bool(any(value > 1e-6 for value in mean_action_delta)),
            "mean_post_onset_executed_action_l2_delta_from_k0": float(np.mean(mean_executed_delta)),
            "mean_post_onset_executed_action_changed": bool(any(value > 1e-6 for value in mean_executed_delta)),
        }
    return report


def run(
    config_path: str,
    checkpoint_path: str,
    query_cases_override: int | None = None,
) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    cutin_inner = config.get("cutin_inner")
    if cutin_inner is None or bool(cutin_inner.get("allow_outer", True)):
        raise ValueError("validation requires the no-Outer Cut-in Inner configuration")
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    if checkpoint.stage != TrainingStage.CONTEXT_META.value:
        raise ValueError("validation requires an interaction_prior → context_meta checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    tasks = select_cutin_validation_tasks(
        load_taskbook(taskbook_path), cutin_inner.get("validation_geometry_ids", ()),
    )
    evaluation = config["evaluation"]
    support_shots = tuple(int(value) for value in evaluation["support_shots"])
    if support_shots != (0, 1, 2, 4):
        raise ValueError("Cut-in Inner validation requires K=0/1/2/4")
    queries = int(evaluation["query_cases"] if query_cases_override is None else query_cases_override)
    if queries < 1:
        raise ValueError("fixed-query validation requires at least one query case")
    max_support = max(support_shots)
    x0 = _fixed_x0(config)
    records: list[dict[str, Any]] = []
    criteria = FailureCriteria.from_config(config["failure"])
    for seed in (int(value) for value in evaluation["seeds"]):
        for task in tasks:
            shared_prior = _shared_prior_episodes(
                model, task, criteria, queries, max_support, seed, x0,
                int(config["training"]["step_budget"]),
            )
            for shots in support_shots:
                records.extend(_evaluate_task(
                    model, task, criteria, shots, queries, max_support, seed, x0,
                    int(config["training"]["step_budget"]), shared_prior,
                ))
    bootstrap_samples = int(evaluation["paired_bootstrap_samples"])
    bootstrap_seed = int(evaluation["paired_bootstrap_seed"])
    paired_risk_improvement = {}
    paired_continuous_risk_improvement = {}
    for shots in support_shots:
        rows = [row for row in records if row["support_shots"] == shots]
        paired_risk_improvement[str(shots)] = paired_bootstrap(
            paired_policy_deltas(rows, "adapted_h_z", "shared_prior"),
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        paired_continuous_risk_improvement[str(shots)] = paired_bootstrap(
            _paired_field_deltas(
                rows, "adapted_h_z", "shared_prior", "continuous_risk_score"
            ),
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    return {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "train",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint_stage": checkpoint.stage,
        "fixed_query_x0": {
            "candidate_index": x0.candidate_index,
            "continuous": list(x0.continuous),
        },
        "support_protocol": {
            "query": "fixed_x0",
            "support": "deterministic_task_local_low_discrepancy_distinct_from_query",
            "support_sets": "nested_prefixes_for_K_1_2_4",
            "adaptation": "posterior_z_inference_only_policy_weights_frozen",
        },
        "support_shots": list(support_shots),
        "query_cases": queries,
        "simulator_seeds": list(evaluation["seeds"]),
        "query_records": records,
        "support_effects": _support_effects(records, support_shots),
        "paired_risk_improvement_adapted_minus_shared_prior": paired_risk_improvement,
        "paired_continuous_risk_improvement_adapted_minus_shared_prior": paired_continuous_risk_improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--query-cases", type=int, default=None,
        help="optional fixed-query replication count; does not alter checkpoint compatibility",
    )
    args = parser.parse_args()
    report = run(args.config, args.checkpoint, args.query_cases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
