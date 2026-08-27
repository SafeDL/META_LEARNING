"""Create current-policy Stage 1 base replays without loading an old checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..failure.criteria import FailureCriteria
from ..scenario.option import AdversarialOption
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.taskbook import load_taskbook
from ..training.pipeline import build_model, load_config, source_tree_provenance
from ..training.trainers import build_online
from .visualize_stage1 import _capture, _write_gif


TASK_IDS = (
    "merge-g04-fast_small_gap",
    "cutin-g04-fast_small_gap",
    "roundabout-g04-fast_small_gap",
)


def _base_scene_action(*_: Any) -> NormalizedScenarioAction:
    return NormalizedScenarioAction(
        0,
        (0.0, 0.0, 0.0, 0.0),
        AdversarialOption.APPROACH_CONFLICT,
    )


def run(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    from panda3d.core import loadPrcFileData

    with tempfile.TemporaryDirectory(prefix="mvr_panda3d_") as cache_dir:
        loadPrcFileData("", f"model-cache-dir {Path(cache_dir).as_posix()}")
        return _run(config_path, output)


def _run(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    settings = dict(config["visualization"])
    topdown = dict(settings["topdown"])
    tail_camera = dict(settings["tail_camera"])
    model = build_model(config, device)
    model.eval()
    criteria = FailureCriteria.from_config(config["failure"])
    tasks_by_id = {task.task_id: task for task in load_taskbook(taskbook_path)}
    try:
        tasks = [tasks_by_id[task_id] for task_id in TASK_IDS]
    except KeyError as error:
        raise ValueError(f"base visualization task is absent from taskbook: {error.args[0]}") from error
    destination = Path(output)
    visualization = destination / "visualization"
    visualization.mkdir(parents=True, exist_ok=True)
    reports = []
    for task in tasks:
        online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
        selected = online.run(
            task,
            1,
            deterministic=True,
            posterior_support_limit=0,
            scene_action_provider=_base_scene_action,
            inner_action_provider=lambda _: np.zeros(3, dtype=np.float32),
        ).episodes[0]
        replay = _capture(
            online,
            task,
            selected.concrete_scenario,
            0,
            "base",
            topdown,
            tail_camera,
        )
        family_dir = visualization / task.functional_scenario
        family_dir.mkdir(exist_ok=True)
        _write_gif(replay.frames, family_dir / "base.gif", int(topdown["fps"]))
        reports.append({
            "family": task.functional_scenario,
            "task_id": task.task_id,
            "scenario": replay.scenario.to_dict(),
            "outcome": dict(replay.outcome),
            "frames": len(replay.frames),
        })
    report = {
        "mode": "stage1_current_lane_stable_base_visualization",
        "media": "gif",
        "policy": "base_zero_interaction_residual",
        "trained_checkpoint": None,
        "control_contract": dict(config["control"]),
        "source_provenance": source_tree_provenance(),
        "families": reports,
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
