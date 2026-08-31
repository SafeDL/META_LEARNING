"""Validate Cut-in Inner adaptation without using a test split or Outer policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluation.fewshot_inner import paired_bootstrap, paired_policy_deltas, valid_critical_score
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


def _provider(action: NormalizedScenarioAction):
    return lambda _task, _index, _candidates, _space: action


def _episode_seed(task: Any, index: int, shots: int, max_support: int, seed: int) -> int:
    source = index if index < shots else max_support + index - shots
    return int(task.geometry_seed + 100_000 * int(seed) + source)


def _first_inner_action(episode: Any) -> list[float]:
    """Capture the first post-onset control, not the pre-onset lockout action."""
    transition = episode.rollout.transitions[0]
    for candidate in episode.rollout.transitions:
        if candidate["info"].get("semantic_maneuver_started", False):
            transition = candidate
            break
    value = transition.get("executed_action", transition["action"])
    return [float(item) for item in np.asarray(value, dtype=float).tolist()]


def _records(
    episodes: Sequence[Any], task: Any, shots: int, seed: int, policy: str,
    x0: NormalizedScenarioAction,
) -> list[dict[str, Any]]:
    return [
        {
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
            "first_inner_action": _first_inner_action(episode),
            "fixed_query_x0": {
                "candidate_index": x0.candidate_index,
                "continuous": list(x0.continuous),
            },
        }
        for query_case_id, episode in enumerate(episodes)
    ]


def _evaluate_task(
    model: Any, task: Any, criteria: FailureCriteria, shots: int, queries: int,
    max_support: int, seed: int, x0: NormalizedScenarioAction, step_budget: int,
) -> list[dict[str, Any]]:
    online = build_online(model, task, step_budget, criteria)
    provider = _provider(x0)
    adapted = online.run(
        task, shots + queries, deterministic=True, posterior_support_limit=shots,
        scene_action_provider=provider,
        episode_seed_provider=lambda current, index: _episode_seed(
            current, index, shots, max_support, seed,
        ),
    ).episodes[shots:]
    shared_prior = online.run(
        task, queries, deterministic=True, posterior_support_limit=0,
        scene_action_provider=provider,
        episode_seed_provider=lambda current, index: _episode_seed(
            current, index, 0, max_support, seed,
        ),
    ).episodes
    return (
        _records(adapted, task, shots, seed, "adapted_h_z", x0)
        + _records(shared_prior, task, shots, seed, "shared_prior", x0)
    )


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
            ))
        z_delta = [row[0] for row in deltas]
        action_delta = [row[1] for row in deltas]
        report[str(shots)] = {
            "pairs": len(deltas),
            "mean_z_l2_delta_from_k0": float(np.mean(z_delta)),
            "mean_first_inner_action_l2_delta_from_k0": float(np.mean(action_delta)),
            "z_changed": bool(any(value > 1e-6 for value in z_delta)),
            "inner_action_changed": bool(any(value > 1e-6 for value in action_delta)),
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
    for shots in support_shots:
        for seed in (int(value) for value in evaluation["seeds"]):
            for task in tasks:
                records.extend(_evaluate_task(
                    model, task, criteria, shots, queries, max_support, seed, x0,
                    int(config["training"]["step_budget"]),
                ))
    bootstrap_samples = int(evaluation["paired_bootstrap_samples"])
    bootstrap_seed = int(evaluation["paired_bootstrap_seed"])
    paired_risk_improvement = {}
    for shots in support_shots:
        rows = [row for row in records if row["support_shots"] == shots]
        paired_risk_improvement[str(shots)] = paired_bootstrap(
            paired_policy_deltas(rows, "adapted_h_z", "shared_prior"),
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
        "support_shots": list(support_shots),
        "query_cases": queries,
        "simulator_seeds": list(evaluation["seeds"]),
        "query_records": records,
        "support_effects": _support_effects(records, support_shots),
        "paired_risk_improvement_adapted_minus_shared_prior": paired_risk_improvement,
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
