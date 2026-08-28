"""Run the G3 SUT heterogeneity validation with shared configurations."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..evaluation.baselines import low_discrepancy_samples
from ..failure.criteria import FailureCriteria
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..sut.registry import default_registry
from ..training.runner import HierarchicalRunner
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import assert_taskbook_compatible, build_model, checkpoint_config_hash, load_config
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.stages import TrainingStage
from ..training.trainers import build_online
from ..validation.gates import gate_sut_heterogeneity


def _actions(space, count: int, seed: int) -> list[NormalizedScenarioAction]:
    samples = low_discrepancy_samples(seed, space.continuous_dim + 1, count)
    return [
        NormalizedScenarioAction(
            min(len(space.candidates) - 1, int((row[0] + 1.0) * 0.5 * len(space.candidates))),
            tuple(float(value) for value in row[1:]),
            space.options[0],
        )
        for row in samples
    ]


def _profile_vector(
    rows: list[dict[str, object]],
    profile_id: str,
    criteria: FailureCriteria,
) -> np.ndarray:
    ordered = sorted(
        (row for row in rows if row["profile"] == profile_id),
        key=lambda row: (str(row["family"]), int(row["config_id"])),
    )
    return np.asarray(
        [
            [
                row["failure"],
                *(np.asarray(row["severity"], dtype=float) / (criteria.severity_bins - 1)),
            ]
            for row in ordered
        ],
        dtype=np.float32,
    )


def run(taskbook: str | Path, criteria: FailureCriteria, *, configurations: int = 16) -> dict[str, object]:
    taskbook_rows = load_taskbook(taskbook)
    adapters = load_adapters()
    templates = {}
    for family in adapters:
        try:
            templates[family] = next(
                task for task in taskbook_rows
                if task.functional_scenario == family and task.geometry_split == "train"
            )
        except StopIteration as error:
            raise ValueError(f"G3 taskbook is missing {family!r}") from error
    registry = default_registry()
    profile_ids = sorted({task.sut_ref for task in taskbook_rows if task.sut_split == "train"})
    profiles = [registry.profiles[profile_id] for profile_id in profile_ids]
    if len(profiles) < 2 or any(profile.adapter_name != "idm" for profile in profiles):
        raise ValueError("G3 requires at least two meta-train IDM profiles")
    spaces = mvr_parameter_spaces()
    executor, runner = ScenarioExecutor(adapters, spaces, registry), HierarchicalRunner(criteria=criteria)
    rows: list[dict[str, object]] = []
    for family, template in templates.items():
        space = spaces[template.functional_scenario]
        for profile in profiles:
            task = replace(template, sut_ref=profile.profile_id)
            for config_id, action in enumerate(_actions(space, configurations, template.geometry_seed)):
                episode = executor.reset(task, action)
                try:
                    rollout = runner.rollout(episode, family, action.option.value, lambda _: np.zeros(2, dtype=np.float32))
                finally:
                    episode.env.close()
                rows.append(
                    {
                        "family": family,
                        "profile": profile.profile_id,
                        "config_id": config_id,
                        "failure": float(rollout.signature.is_failure),
                        "valid": float(rollout.signature.is_valid_episode),
                        "severity": list(rollout.signature.severity_vector),
                    }
                )
    vectors = {profile_id: _profile_vector(rows, profile_id, criteria) for profile_id in profile_ids}
    failure_disagreement, severity_distance, overlaps = {}, {}, {}
    for index, left in enumerate(profile_ids):
        for right in profile_ids[index + 1:]:
            key = f"{left}__{right}"
            left_failure, right_failure = vectors[left][:, 0] > 0.5, vectors[right][:, 0] > 0.5
            failure_disagreement[key] = float(np.mean(left_failure != right_failure))
            severity_distance[key] = float(np.sqrt(np.mean((vectors[left][:, 1:] - vectors[right][:, 1:]) ** 2)))
            overlaps[key] = float(np.sum(left_failure & right_failure) / max(1, np.sum(left_failure | right_failure)))
    valid_rate = float(np.mean([row["valid"] for row in rows]))
    failure_rate = float(np.mean([row["failure"] for row in rows]))
    disagreement = float(np.mean(list(failure_disagreement.values())))
    severity = float(np.mean(list(severity_distance.values())))
    return {
        "configurations_per_family": configurations,
        "episodes": len(rows),
        "profiles": list(vectors),
        "rows": rows,
        "pairwise_failure_disagreement": failure_disagreement,
        "pairwise_severity_distance": severity_distance,
        "dangerous_region_overlap": overlaps,
        "mean_failure_disagreement": disagreement,
        "mean_severity_distance": severity,
        "gate": gate_sut_heterogeneity(disagreement, severity, valid_rate, failure_rate),
    }


def run_inner_validation(
    config: dict[str, Any],
    taskbook: str | Path,
    checkpoint_path: str | Path,
    device: Any,
    criteria: FailureCriteria,
    *,
    cases_per_task: int = 16,
) -> dict[str, object]:
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    if checkpoint.stage != TrainingStage.INNER_PRETRAIN.value:
        raise ValueError("Inner validation requires an inner_pretrain checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    # Stage 1 compares a fixed policy mean on identical initial conditions.
    # The general evaluation flag controls the later stochastic outer protocol.
    deterministic = True
    family = str(config["training"].get("family_filter", "all"))
    tasks = [
        task for task in load_taskbook(taskbook)
        if task.sut_split == "validation"
        and task.geometry_split == "validation"
        and task.functional_split == "train"
        and (family == "all" or task.functional_scenario == family)
    ]
    if not tasks:
        raise ValueError("taskbook has no validation SUT + validation geometry tasks")

    def evaluate_policy(name: str, action_provider: Any = None) -> dict[str, object]:
        sampler = PretrainSceneSampler(tuple(tasks), cases_per_task, int(config["seed"]))
        records: list[dict[str, object]] = []
        for task in tasks:
            result = build_online(
                model, task, int(config["training"]["step_budget"]), criteria
            ).run(
                task,
                cases_per_task,
                deterministic=deterministic,
                posterior_support_limit=0,
                scene_action_provider=sampler,
                inner_action_provider=action_provider,
            )
            for episode in result.episodes:
                records.append(
                    {
                        "family": task.functional_scenario,
                        "signature": episode.rollout.signature,
                        "outcome": episode.rollout.outcome,
                        "episode_return": sum(
                            float(row["reward_inner"])
                            for row in episode.rollout.transitions
                        ),
                    }
                )

        def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
            signatures = [row["signature"] for row in rows]
            outcomes = [row["outcome"] for row in rows]
            valid = [float(signature.is_valid_episode) for signature in signatures]
            failures = [float(signature.is_failure) for signature in signatures]
            violation_fields = (
                "non_target_collision",
                "adversary_out_of_road",
                "sut_out_of_road",
                "wrong_route",
                "adversary_traffic_violation",
            )
            return {
                "episodes": len(rows),
                "valid_rate": float(np.mean(valid)),
                "invalid_rate": float(1.0 - np.mean(valid)),
                "valid_critical_rate": float(np.mean(failures)),
                "critical_rate_given_valid": float(
                    np.sum(failures) / max(np.sum(valid), 1.0)
                ),
                "target_collision_rate": float(
                    np.mean([float(row.get("valid_target_collision", False)) for row in outcomes])
                ),
                "invalid_target_collision_rate": float(
                    np.mean([
                        float(row.get("target_collision", False) and not row.get("is_valid_episode", False))
                        for row in outcomes
                    ])
                ),
                "near_miss_rate": float(
                    np.mean([float(row.get("valid_critical_near_miss", False)) for row in outcomes])
                ),
                "median_min_ttc": float(
                    np.median([float(row.get("min_ttc", 0.0)) for row in outcomes])
                ),
                "median_min_distance": float(
                    np.median([float(row.get("min_distance", 0.0)) for row in outcomes])
                ),
                "max_closing_speed": float(
                    np.max([float(row.get("max_closing_speed", 0.0)) for row in outcomes])
                ),
                "mean_inner_episode_return": float(
                    np.mean([float(row["episode_return"]) for row in rows])
                ),
                "violation_rates": {
                    field: float(np.mean([float(row["outcome"].get(field, False)) for row in rows]))
                    for field in violation_fields
                },
            }

        by_family = {
            family: summarize([row for row in records if row["family"] == family])
            for family in sorted({str(row["family"]) for row in records})
        }
        return {
            "policy": name,
            **summarize(records),
            "by_family": by_family,
        }

    random_rng = np.random.default_rng(int(config["seed"]) + 1)
    reports = [
        evaluate_policy("base", lambda _: np.zeros(2, dtype=np.float32)),
        evaluate_policy("random_residual", lambda _: random_rng.uniform(-1.0, 1.0, 2).astype(np.float32)),
        evaluate_policy("trained_inner"),
    ]
    return {
        "mode": "inner",
        "regime": "validation_sut_validation_geometry",
        "deterministic": deterministic,
        "cases_per_task": cases_per_task,
        "tasks": [task.task_id for task in tasks],
        "policies": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/mvr.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=("g3", "inner"), default="g3")
    parser.add_argument("--cases-per-task", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "inner":
        if not args.checkpoint:
            parser.error("--checkpoint is required for --mode inner")
        _, taskbook, device = load_config(args.config)
        report = run_inner_validation(
            config,
            taskbook,
            args.checkpoint,
            device,
            FailureCriteria.from_config(config["failure"]),
            cases_per_task=args.cases_per_task,
        )
    else:
        report = run(config["taskbook"], FailureCriteria.from_config(config["failure"]))
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
