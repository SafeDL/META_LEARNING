"""Render trained Cut-in Inner SAC rollouts from an interaction-prior checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..experiments.cutin_inner import expand_cutin_training_domains
from ..failure.criteria import FailureCriteria
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
    selected_tasks,
    seed_everything,
)
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.stages import TrainingStage
from ..training.trainers import build_online
from .render_cutin_inner_policy_gif import (
    VISUAL_ENVIRONMENT_OVERRIDES,
    _capture_frames,
    _save_gif,
)


def _training_tasks(config: dict[str, Any], taskbook_path: Path) -> list[Any]:
    tasks = selected_tasks(config, taskbook_path, "train", "train", "train")
    settings = config["cutin_inner"]
    allowed_suts = {str(value) for value in settings["training_sut_refs"]}
    allowed_geometries = {str(value) for value in settings["training_geometry_ids"]}
    tasks = [
        task for task in tasks
        if task.sut_ref in allowed_suts and task.geometry_id in allowed_geometries
    ]
    return expand_cutin_training_domains(tasks, settings["training_logical_domains"])


def _decision_rows(episode: Any) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "raw_policy_action": [
                float(value) for value in row["raw_policy_action"]
            ],
            "planner_action": [float(value) for value in row["planner_action"]],
            "executed_vehicle_action": [
                float(value) for value in row["executed_vehicle_action"]
            ],
        }
        for index, row in enumerate(episode.rollout.transitions)
        if row["info"].get("inner_policy_decision", False)
    ]


def _domain_label(task: Any) -> str:
    domain = str(task.logical_domain_id).replace("_", " ")
    return f"trained Inner SAC | {domain}"


def run(config_path: str, checkpoint_path: str, output_dir: str) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    if checkpoint.stage != TrainingStage.INTERACTION_PRIOR.value:
        raise ValueError("trained Cut-in GIF rendering requires an interaction_prior checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    seed_everything(int(config["seed"]))
    tasks = _training_tasks(config, taskbook_path)
    if not tasks:
        raise ValueError("Cut-in trained GIF rendering requires training tasks")
    domains = [str(task.logical_domain_id) for task in tasks]
    if len(domains) != len(set(domains)):
        raise ValueError("Cut-in trained GIF rendering requires unique logical domains")
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    episodes_per_task = int(config["interaction_prior"]["episodes_per_task"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        sampler = PretrainSceneSampler((task,), episodes_per_task, int(config["seed"]))
        online = build_online(
            model, task, int(config["training"]["step_budget"]),
            FailureCriteria.from_config(config["failure"]),
        )
        frames, callback = _capture_frames(
            _domain_label(task), action_key="planner_action"
        )
        result = online.run(
            task,
            1,
            deterministic=True,
            posterior_support_limit=0,
            episode_index_offset=0,
            scene_action_provider=sampler,
            rollout_step_callback=callback,
            environment_overrides=VISUAL_ENVIRONMENT_OVERRIDES,
        )
        episode = result.episodes[0]
        path = output / f"{task.logical_domain_id}.gif"
        _save_gif(frames, path)
        rows.append({
            "gif": str(path),
            "logical_domain_id": task.logical_domain_id,
            "candidate": episode.concrete_scenario.candidate_id,
            "logical_parameters": dict(episode.concrete_scenario.initial_state),
            "frames": len(frames),
            "policy_decisions": _decision_rows(episode),
            "cutin_completed": any(
                row["info"].get("semantic_maneuver_completed", False)
                for row in episode.rollout.transitions
            ),
            "outcome": dict(episode.outcome),
        })
    report = {
        "scope": "deterministic trained Inner SAC policy replayed in actual Cut-in training tasks",
        "checkpoint": str(checkpoint_path),
        "task_id": task.task_id,
        "rollouts": rows,
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
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
