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
    samples = sobol_like(seed, space.continuous_dim + 2, count)
    result = []
    for row in samples:
        candidate = min(len(space.candidates) - 1, int((row[0] + 1.0) * 0.5 * len(space.candidates)))
        option = space.options[min(len(space.options) - 1, int((row[1] + 1.0) * 0.5 * len(space.options)))]
        result.append(NormalizedScenarioAction(candidate, tuple(float(value) for value in row[2:]), option))
    return result


def run(taskbook: str | Path, *, configurations: int = 16) -> dict[str, object]:
    tasks = {task.scenario_family: task for task in load_taskbook(taskbook)}
    if set(tasks) != set(ADAPTERS):
        raise ValueError("Gate A taskbook must contain exactly one task for merge, cutin, and roundabout")
    registry = default_registry()
    profiles = [profile for profile in registry.profiles.values() if profile.adapter_name == "idm"]
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
        vector = [next(row for row in rows if row["profile"] == profile.profile_id and row["family"] == family and row["config_id"] == config_id) for family in ADAPTERS for config_id in range(configurations)]
        vectors[profile.profile_id] = np.asarray([[row["failure"], 1.0 - row["valid"], *(np.asarray(row["severity"], dtype=float) / 4.0)] for row in vector])
    names = list(vectors)
    pairwise = {f"{left}__{right}": float(np.linalg.norm(vectors[left] - vectors[right]) / np.sqrt(vectors[left].size)) for index, left in enumerate(names) for right in names[index + 1:]}
    rng, within = np.random.default_rng(0), []
    for vector in vectors.values():
        for _ in range(128):
            left, right = rng.integers(0, len(vector), size=(2, len(vector)))
            within.append(float(np.linalg.norm(vector[left].mean(0) - vector[right].mean(0))))
    valid_rate = float(np.mean([row["valid"] for row in rows]))
    failure_rate = float(np.mean([row["failure"] for row in rows]))
    gate = gate_failure_landscape(float(np.mean(within)), float(np.mean(list(pairwise.values()))), valid_rate, failure_rate)
    overlaps = {key: float(np.sum((vectors[left][:, 0] > 0) & (vectors[right][:, 0] > 0)) / max(1, np.sum((vectors[left][:, 0] > 0) | (vectors[right][:, 0] > 0)))) for key in pairwise for left, right in [key.split("__", 1)]}
    return {"configurations_per_family": configurations, "episodes": len(rows), "profiles": names, "rows": rows, "pairwise_distance": pairwise, "dangerous_region_overlap": overlaps, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.taskbook), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
