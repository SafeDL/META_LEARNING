"""Build a Base-safe casebook independently from residual reachability probes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.headroom_casebook import HeadroomCasebook, is_base_safe_headroom
from ..training.pipeline import build_model, load_config
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.trainers import build_online


def run(config_path: str, output: str) -> HeadroomCasebook:
    config, taskbook_path, device = load_config(config_path)
    settings = dict(config["headroom"])
    cases = int(settings["cases_per_task"])
    candidate_pool = cases * int(settings.get("candidate_pool_multiplier", 4))
    calibration_sut_split = str(settings.get("calibration_sut_split", "validation"))
    logical_split = str(settings.get("logical_split", "test"))
    geometry_split = str(settings.get("geometry_split", "train"))
    tasks = [task for task in load_taskbook(taskbook_path) if task.sut_split == calibration_sut_split and task.logical_split == logical_split and task.geometry_split == geometry_split]
    if not tasks:
        raise ValueError("headroom calibration task selection is empty")
    model = build_model(config, device)
    model.eval()
    sampler = PretrainSceneSampler(tuple(tasks), candidate_pool, int(config["seed"]))
    criteria = FailureCriteria.from_config(config["failure"])
    actions = {}
    for task in tasks:
        online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
        result = online.run(
            task, candidate_pool, deterministic=True, posterior_support_limit=0,
            scene_action_provider=sampler,
            inner_action_provider=lambda _: np.zeros(2, dtype=np.float32),
        )
        accepted = []
        for episode in result.episodes:
            challenge_steps = sum(bool(row["info"].get("semantic_challenge_phase_active", False)) for row in episode.rollout.transitions)
            if is_base_safe_headroom(episode.outcome, challenge_steps):
                accepted.append(episode.concrete_scenario.replay_action(
                    online.executor.spaces[task.functional_scenario]
                ))
        if len(accepted) < cases:
            raise RuntimeError(f"{task.task_id} has {len(accepted)}/{cases} Base-safe headroom cases")
        actions[task.task_id] = tuple(accepted[:cases])
    calibration_by_key = {
        (task.functional_scenario, task.geometry_id, task.logical_domain_id): values
        for task_id, values in actions.items()
        for task in tasks if task.task_id == task_id
    }
    all_tasks = load_taskbook(taskbook_path)
    for task in all_tasks:
        key = (task.functional_scenario, task.geometry_id, task.logical_domain_id)
        if key in calibration_by_key:
            actions.setdefault(task.task_id, calibration_by_key[key])
    casebook = HeadroomCasebook(actions, {
        "cases_per_task": cases,
        "candidate_pool": candidate_pool,
        "calibration_sut_split": calibration_sut_split,
        "logical_split": logical_split,
        "geometry_split": geometry_split,
        "base_policy": "zero_residual",
    })
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    casebook.save(output)
    return casebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/mvr.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
