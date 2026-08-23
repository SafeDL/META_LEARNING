"""Run the G3 SUT heterogeneity validation with shared configurations."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

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
        space = spaces[template.functional_scenario + "_v1"]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/mvr.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = run(config["taskbook"], FailureCriteria.from_config(config["failure"]))
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
