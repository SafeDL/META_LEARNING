"""No-learning residual reachability probe, separate from calibration-casebook evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.pipeline import build_model, load_config
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.trainers import build_online


RESIDUALS = {
    "base": (0.0, 0.0),
    "steer_left": (-0.5, 0.0),
    "steer_right": (0.5, 0.0),
    "brake": (0.0, -0.75),
    "accelerate": (0.0, 0.75),
}


def run(config_path: str, output: str) -> dict[str, object]:
    config, taskbook_path, device = load_config(config_path)
    cases = int(config.get("reachability", {}).get("cases_per_task", 4))
    tasks = [task for task in load_taskbook(taskbook_path) if task.sut_split == "validation" and task.logical_split == "validation" and task.geometry_split == "train"]
    if not tasks:
        raise ValueError("reachability requires validation SUT/domain tasks")
    model = build_model(config, device)
    model.eval()
    sampler = PretrainSceneSampler(tuple(tasks), cases, int(config["seed"]))
    criteria = FailureCriteria.from_config(config["failure"])
    rows = []
    for task in tasks:
        online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
        for name, residual in RESIDUALS.items():
            result = online.run(
                task, cases, deterministic=True, posterior_support_limit=0,
                scene_action_provider=sampler,
                inner_action_provider=lambda _, value=residual: np.asarray(value, dtype=np.float32),
            )
            for episode in result.episodes:
                rows.append({
                    "task_id": task.task_id,
                    "family": task.functional_scenario,
                    "residual": name,
                    "valid": bool(episode.outcome["is_valid_episode"]),
                    "critical": bool(episode.outcome["is_failure"]),
                    "challenge_steps": sum(bool(step["info"].get("semantic_challenge_phase_active", False)) for step in episode.rollout.transitions),
                })
    report = {"scope": "no_learning_residual_reachability", "residuals": RESIDUALS, "rows": rows}
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/mvr.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
