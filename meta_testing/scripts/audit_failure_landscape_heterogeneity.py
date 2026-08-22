"""Run the pure-IDM Gate A failure-landscape audit."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ..audits import gate_failure_landscape
from ..evaluation.baselines import sobol_like
from ..scenario.adapters import CutInScenarioAdapter, MergeScenarioAdapter, RoundaboutScenarioAdapter
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.taskbook import load_taskbook
from ..sut.registry import default_registry
from ..training.runner import HierarchicalRunner


ADAPTERS = {"merge": MergeScenarioAdapter(), "cutin": CutInScenarioAdapter(), "roundabout": RoundaboutScenarioAdapter()}


def _actions(space, count: int, seed: int) -> list[NormalizedScenarioAction]:
    samples = sobol_like(seed, space.continuous_dim + 1, count)
    result = []
    for row in samples:
        candidate = min(len(space.candidates) - 1, int((row[0] + 1.0) * 0.5 * len(space.candidates)))
        result.append(NormalizedScenarioAction(candidate, tuple(float(value) for value in row[1:]), space.options[0]))
    return result


def run(taskbook: str | Path, *, configurations: int = 16) -> dict[str, object]:
    taskbook_rows = load_taskbook(taskbook)
    tasks = {}
    for family in ADAPTERS:
        family_tasks = [task for task in taskbook_rows if task.scenario_family == family]
        if not family_tasks:
            raise ValueError(f"Gate A taskbook is missing {family!r}")
        tasks[family] = family_tasks[0]

    train_profile_ids = sorted({task.sut_ref for task in taskbook_rows if task.split == "meta_train"})
    registry = default_registry()
    profiles = [registry.profiles[profile_id] for profile_id in train_profile_ids]
    if len(profiles) < 2 or any(profile.adapter_name != "idm" for profile in profiles):
        raise ValueError("Gate A requires at least two meta-train IDM profiles")
    spaces = mvr_parameter_spaces()
    executor, runner = ScenarioExecutor(ADAPTERS, spaces, registry), HierarchicalRunner()
    rows: list[dict[str, object]] = []
    for family, template in tasks.items():
        space = spaces[template.parameter_space_id]
        for profile in profiles:
            task = replace(template, sut_ref=profile.profile_id)
            for config_id, action in enumerate(_actions(space, configurations, template.seed)):
                episode = executor.reset(task, action)
                try:
                    rollout = runner.rollout(episode, family, action.option.value, lambda _: np.zeros(2, dtype=np.float32))
                finally:
                    episode.env.close()
                signature = rollout.signature
                rows.append({"family": family, "profile": profile.profile_id, "config_id": config_id, "failure": float(signature.is_failure), "valid": float(signature.is_valid_episode), "severity": list(signature.severity_vector)})
    vectors: dict[str, np.ndarray] = {}
    for profile in profiles:
        rows_for_profile = [
            next(
                row for row in rows
                if row["profile"] == profile.profile_id
                and row["family"] == family
                and row["config_id"] == config_id
            )
            for family in ADAPTERS
            for config_id in range(configurations)
        ]
        vectors[profile.profile_id] = np.asarray(
            [
                [row["failure"], *(np.asarray(row["severity"], dtype=float) / 4.0)]
                for row in rows_for_profile
            ],
            dtype=np.float32,
        )
    names = list(vectors)
    failure_disagreement: dict[str, float] = {}
    severity_distance: dict[str, float] = {}
    overlaps: dict[str, float] = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            key = f"{left}__{right}"
            left_failure = vectors[left][:, 0] > 0.5
            right_failure = vectors[right][:, 0] > 0.5
            failure_disagreement[key] = float(np.mean(left_failure != right_failure))
            severity_distance[key] = float(np.sqrt(np.mean((vectors[left][:, 1:] - vectors[right][:, 1:]) ** 2)))
            overlaps[key] = float(np.sum(left_failure & right_failure) / max(1, np.sum(left_failure | right_failure)))
    valid_rate = float(np.mean([row["valid"] for row in rows]))
    failure_rate = float(np.mean([row["failure"] for row in rows]))
    mean_failure_disagreement = float(np.mean(list(failure_disagreement.values())))
    mean_severity_distance = float(np.mean(list(severity_distance.values())))
    gate = gate_failure_landscape(mean_failure_disagreement, mean_severity_distance, valid_rate, failure_rate)
    return {
        "configurations_per_family": configurations,
        "episodes": len(rows),
        "profiles": names,
        "rows": rows,
        "pairwise_failure_disagreement": failure_disagreement,
        "pairwise_severity_distance": severity_distance,
        "dangerous_region_overlap": overlaps,
        "mean_failure_disagreement": mean_failure_disagreement,
        "mean_severity_distance": mean_severity_distance,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.taskbook), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
