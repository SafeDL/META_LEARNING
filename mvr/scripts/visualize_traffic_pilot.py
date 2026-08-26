"""Render one fixed lawful pilot GIF for every Stage 1 scenario family."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.option import AdversarialOption
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..safety import TrafficActionShield
from ..training.pipeline import load_config
from ..training.runner import HierarchicalRunner
from .visualize_stage1 import (
    ADVERSARY_COLOR,
    SUT_COLOR,
    VISUAL_ENVIRONMENT_OVERRIDES,
    _chase_frame,
    _dual_view_frame,
    _paint_role,
    _write_gif,
)


OUTPUT_DIRECTORY = Path("results/mvr/stage1_traffic_pilot")


@dataclass(frozen=True)
class PilotCase:
    family: str
    task_id: str
    candidate_index: int
    continuous: tuple[float, float, float, float]
    max_steps: int


PILOT_CASES = (
    PilotCase("merge", "merge-g04-fast_small_gap", 0, (0.0, 0.0, 0.0, 0.0), 60),
    PilotCase("cutin", "cutin-g04-fast_small_gap", 0, (-1.0, 1.0, 0.0, 0.0), 60),
    PilotCase("roundabout", "roundabout-g04-fast_small_gap", 0, (0.0, 0.0, -1.0, -1.0), 60),
)


def _task(case: PilotCase) -> Any:
    return next(
        task for task in load_taskbook("mvr/configs/taskbook.json") if task.task_id == case.task_id
    )


def _traffic_status(episode: Any, info: Mapping[str, Any]) -> str:
    if info.get("traffic_shield_rejected"):
        status = str(info.get("traffic_shield_rejection_reason"))
    elif info.get("traffic_lane_change_completed"):
        status = "legal merge complete"
    elif info.get("traffic_lane_change_started"):
        status = "legal merge active"
    else:
        status = "lane following"
    route_progress = episode.adversary_route.projection(
        episode.adversary.position,
        episode.adversary.heading_theta,
    ).s_m
    return f"{status} | route: {route_progress:.0f}/{episode.adversary_route.length_m:.0f} m"


def _inner_action(episode: Any) -> Any:
    contract = episode.layout.traffic_contract
    if contract.target_lane_number is None:
        return lambda _: np.zeros(2, dtype=np.float32)
    target_lane = episode.env.current_map.road_network.get_lane(
        (*episode.layout.adversary_lane[:2], contract.target_lane_number)
    )
    steering = float(np.sign(TrafficActionShield._lane_follow_action(episode.adversary, target_lane)))
    return lambda _: np.asarray((steering, 0.0), dtype=np.float32)


def _capture(case: PilotCase) -> tuple[list[np.ndarray], Mapping[str, Any]]:
    config, _, _ = load_config("mvr/configs/mvr_stage1.yaml")
    visualization = dict(config["visualization"])
    topdown = dict(visualization["topdown"])
    tail_camera = dict(visualization["tail_camera"])
    task = _task(case)
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    episode = executor.reset(
        task,
        NormalizedScenarioAction(
            case.candidate_index,
            case.continuous,
            AdversarialOption.APPROACH_CONFLICT,
        ),
        episode_seed=204,
        environment_overrides={
            **VISUAL_ENVIRONMENT_OVERRIDES,
            "camera_height": float(tail_camera["height"]),
            "camera_dist": float(tail_camera["distance"]),
            "camera_pitch": float(tail_camera["pitch"]),
            "camera_smooth": False,
        },
    )
    frames: list[np.ndarray] = []

    def callback(current: Any, step: int, info: Mapping[str, Any]) -> None:
        camera = current.env.engine.main_camera
        if step == 0:
            camera.track(current.sut)
            _paint_role(current.sut, SUT_COLOR)
            _paint_role(current.adversary, ADVERSARY_COLOR)
        topdown_frame = current.env.render(
            mode="topdown",
            window=False,
            screen_size=tuple(int(value) for value in topdown["screen_size"]),
            scaling=int(topdown["scaling"]),
            camera_position=current.layout.conflict_xy,
        )
        frames.append(
            _dual_view_frame(
                _chase_frame(current),
                np.asarray(topdown_frame),
                f"shielded {case.family} pilot",
                current.layout.candidate,
                step,
                _traffic_status(current, info),
            )
        )

    try:
        rollout = HierarchicalRunner(max_steps=case.max_steps).rollout(
            episode,
            case.family,
            AdversarialOption.APPROACH_CONFLICT.value,
            _inner_action(episode),
            step_callback=callback,
        )
        if not frames:
            raise RuntimeError("traffic-pilot renderer produced no frames")
        if not rollout.outcome["is_valid_episode"]:
            raise RuntimeError(f"{case.family} pilot produced a traffic violation")
        return frames, {
            "task_id": task.task_id,
            "candidate": episode.layout.candidate,
            "outcome": dict(rollout.outcome),
        }
    finally:
        episode.env.close()


def run(output: Path = OUTPUT_DIRECTORY) -> Mapping[str, Any]:
    """Write three independent 15 FPS GIFs after the focused pilots pass."""
    from panda3d.core import loadPrcFileData

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mvr_panda3d_") as cache_dir:
        loadPrcFileData("", f"model-cache-dir {Path(cache_dir).as_posix()}")
        reports = []
        for case in PILOT_CASES:
            frames, report = _capture(case)
            filename = f"{case.family}.gif"
            _write_gif(frames, output / filename, fps=15)
            reports.append({"family": case.family, "gif": filename, "frames": len(frames), **report})
    manifest = {"mode": "fixed_lawful_stage1_pilot", "media": "gif", "cases": reports}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = run()
    print(json.dumps({
        "mode": manifest["mode"],
        "media": manifest["media"],
        "cases": [
            {
                "family": case["family"],
                "gif": case["gif"],
                "frames": case["frames"],
                "is_valid_episode": case["outcome"]["is_valid_episode"],
                "adversary_traffic_violation": case["outcome"]["adversary_traffic_violation"],
            }
            for case in manifest["cases"]
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
