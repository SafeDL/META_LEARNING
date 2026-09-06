"""Compare an interaction-prior Inner SAC with a random planner on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.catalog import mvr_parameter_spaces
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
from .render_cutin_inner_policy_gif import (
    VISUAL_ENVIRONMENT_OVERRIDES,
    _capture_frames,
    _save_gif,
)


DEFAULT_CASE_COUNT = 20
DEFAULT_RANDOM_SEED = 20260906


def random_cases(task: Any, count: int, seed: int) -> list[NormalizedScenarioAction]:
    """Draw reproducible paired validation scenarios inside task-local bounds."""
    if count < 1:
        raise ValueError("validation case count must be positive")
    bounds = np.asarray(list(task.logical_domain_bounds.values()), dtype=float)
    generator = np.random.default_rng(seed)
    candidates = len(mvr_parameter_spaces()[task.functional_scenario].candidates)
    return [
        NormalizedScenarioAction(
            int(generator.integers(candidates)),
            tuple(float(value) for value in generator.uniform(bounds[:, 0], bounds[:, 1])),
        )
        for _ in range(count)
    ]


def random_inner_policy(seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a reproducible random 4-D Frenet planner baseline."""
    generator = np.random.default_rng(seed)

    def policy(_state: np.ndarray) -> np.ndarray:
        return generator.uniform(-1.0, 1.0, size=4).astype(np.float32)

    return policy


def _event(outcome: Mapping[str, Any]) -> bool:
    return bool(
        outcome["valid_target_collision"]
        or outcome["valid_critical_near_miss"]
    )


def _record(
    case_index: int,
    policy: str,
    action: NormalizedScenarioAction,
    episode_seed: int,
    episode: Any,
) -> dict[str, Any]:
    outcome = episode.outcome
    transitions = episode.rollout.transitions
    return {
        "case_index": case_index,
        "policy": policy,
        "candidate_index": action.candidate_index,
        "normalized_parameters": list(action.continuous),
        "episode_seed": episode_seed,
        "valid": bool(outcome["is_valid_episode"]),
        "failure": bool(outcome["is_failure"]),
        "event": _event(outcome),
        "target_collision": bool(outcome["valid_target_collision"]),
        "critical_near_miss": bool(outcome["valid_critical_near_miss"]),
        "min_ttc": float(outcome["min_ttc"]),
        "min_distance": float(outcome["min_distance"]),
        "termination_reason": outcome["termination_reason"],
        "cutin_completed": any(
            bool(row["info"].get("semantic_maneuver_completed", False))
            for row in transitions
        ),
        "raw_near_miss_candidate_steps": sum(
            bool(row["info"].get("raw_near_miss_candidate", False))
            for row in transitions
        ),
        "valid_event_capture_steps": sum(
            bool(row["info"].get("event_just_captured", False))
            and (
                bool(row["info"].get("valid_target_collision", False))
                or bool(row["info"].get("valid_critical_near_miss", False))
            )
            for row in transitions
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize public failure contracts rather than raw TTC alone."""
    if not rows:
        raise ValueError("validation summary requires at least one record")
    count = len(rows)
    return {
        "cases": count,
        "valid_rate": float(np.mean([row["valid"] for row in rows])),
        "failure_count": sum(bool(row["failure"]) for row in rows),
        "failure_rate": float(np.mean([row["failure"] for row in rows])),
        "event_count": sum(bool(row["event"]) for row in rows),
        "event_rate": float(np.mean([row["event"] for row in rows])),
        "target_collision_count": sum(bool(row["target_collision"]) for row in rows),
        "critical_near_miss_count": sum(
            bool(row["critical_near_miss"]) for row in rows
        ),
        "raw_near_miss_candidate_steps": sum(
            int(row["raw_near_miss_candidate_steps"]) for row in rows
        ),
        "valid_event_capture_steps": sum(
            int(row["valid_event_capture_steps"]) for row in rows
        ),
    }


def select_render_case_ids(rows: list[dict[str, Any]]) -> list[int]:
    """Choose up to two paired cases with the strongest policy differences."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["case_index"]), []).append(row)
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("render selection requires exactly two policies per case")

    def priority(values: list[dict[str, Any]]) -> tuple[float, float]:
        sac, random = sorted(values, key=lambda value: value["policy"])
        disagreement = float(bool(sac["failure"]) != bool(random["failure"]))
        risk = min(
            min(float(value["min_ttc"]) / 5.0, float(value["min_distance"]) / 10.0)
            for value in values
        )
        return disagreement, -risk

    ordered = sorted(grouped.items(), key=lambda item: priority(item[1]), reverse=True)
    return [case_index for case_index, _ in ordered[:2]]


def _episode(
    online: Any,
    task: Any,
    action: NormalizedScenarioAction,
    episode_seed: int,
    *,
    random_seed: int | None = None,
    callback: Callable[..., None] | None = None,
) -> Any:
    return online.run(
        task,
        1,
        deterministic=True,
        posterior_support_limit=0,
        scene_action_provider=lambda *_args, value=action: value,
        inner_action_provider=(
            random_inner_policy(random_seed) if random_seed is not None else None
        ),
        episode_seed_provider=lambda *_args, value=episode_seed: value,
        rollout_step_callback=callback,
        environment_overrides=(VISUAL_ENVIRONMENT_OVERRIDES if callback else None),
    ).episodes[0]


def run(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    *,
    cases: int = DEFAULT_CASE_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    if checkpoint.stage != TrainingStage.INTERACTION_PRIOR.value:
        raise ValueError("validation comparison requires an interaction_prior checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    tasks = select_cutin_validation_tasks(load_taskbook(taskbook_path))
    if len(tasks) != 1:
        raise ValueError("random validation comparison requires one validation task")
    task = tasks[0]
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    criteria = FailureCriteria.from_config(config["failure"])
    online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
    actions = random_cases(task, cases, random_seed)
    records: list[dict[str, Any]] = []
    for case_index, action in enumerate(actions):
        episode_seed = int(task.geometry_seed + 100_000 + case_index)
        learned = _episode(online, task, action, episode_seed)
        records.append(_record(case_index, "sac", action, episode_seed, learned))
        random = _episode(
            online, task, action, episode_seed, random_seed=random_seed + case_index
        )
        records.append(_record(case_index, "random", action, episode_seed, random))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_render_case_ids(records)
    gifs = []
    for case_index in selected:
        action = actions[case_index]
        episode_seed = int(task.geometry_seed + 100_000 + case_index)
        for policy, policy_seed in (("sac", None), ("random", random_seed + case_index)):
            frames, callback = _capture_frames(f"validation | {policy}")
            episode = _episode(
                online, task, action, episode_seed,
                random_seed=policy_seed, callback=callback,
            )
            path = output / f"case_{case_index:02d}_{policy}.gif"
            _save_gif(frames, path)
            gifs.append({
                "case_index": case_index,
                "policy": policy,
                "path": str(path),
                "frames": len(frames),
                "outcome": dict(episode.outcome),
            })
    report = {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "validation",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint": str(checkpoint_path),
        "task_id": task.task_id,
        "random_scene_count": cases,
        "random_seed": random_seed,
        "paired_protocol": (
            "Each policy receives the identical Logical action, candidate, and episode seed."
        ),
        "random_baseline": "independent uniform 4-D Frenet planner action per replan",
        "summary": {
            "sac": summarize([row for row in records if row["policy"] == "sac"]),
            "random": summarize([row for row in records if row["policy"] == "random"]),
        },
        "records": records,
        "rendered_case_indices": selected,
        "gifs": gifs,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    run(
        args.config, args.checkpoint, args.output_dir,
        cases=args.cases, random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
