"""Render a deterministic, non-bank replay of one observed DIVA source episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from ..diva.behavior import DivaCutInBehavior
from ..diva.episode_executor import BEHAVIOR_CONTRACT
from ..diva.types import DIVA_SCHEMA, DivaCutInDesign
from ..evaluation.fewshot_inner import valid_critical_score
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner
from .render_cutin_inner_policy_gif import (
    ADVERSARY_COLOR,
    SUT_COLOR,
    VISUAL_ENVIRONMENT_OVERRIDES,
    _chase_frame,
    _draw_reference_overlay,
    _paint_role,
)


def _read_row(path: str, design_id: str) -> Mapping[str, Any]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["design"]["design_id"] == design_id:
            return row
    raise ValueError("design id is not present in the supplied DIVA observations")


def _panel(image: np.ndarray, title: str, subtitle: str) -> Image.Image:
    frame = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    canvas = ImageDraw.Draw(frame)
    canvas.rectangle((0, 0, frame.width, 50), fill=(25, 25, 25))
    canvas.text((10, 7), title, fill=(255, 255, 255))
    canvas.text((10, 28), subtitle, fill=(220, 220, 220))
    return frame


def _frame(episode: Any, info: Mapping[str, Any], step: int, label: str) -> np.ndarray:
    topdown = episode.env.render(
        mode="topdown",
        window=False,
        screen_size=(700, 700),
        scaling=5,
        camera_position=episode.layout.conflict_xy,
    )
    points = info["maneuver_reference_points_xy"]
    chase = _draw_reference_overlay(_chase_frame(episode), episode, points, view="chase")
    topdown = _draw_reference_overlay(np.asarray(topdown), episode, points, view="topdown")
    height = min(chase.shape[0], topdown.shape[0])
    chase_image = Image.fromarray(chase).resize((round(chase.shape[1] * height / chase.shape[0]), height))
    topdown_image = Image.fromarray(topdown).resize((round(topdown.shape[1] * height / topdown.shape[0]), height))
    action = info.get("raw_policy_action", (0.0, 0.0, 0.0, 0.0))
    status = "active" if info.get("semantic_maneuver_active") else "lane following"
    if info.get("semantic_maneuver_completed"):
        status = "maneuver complete"
    left = _panel(
        np.asarray(chase_image),
        "Tail view | blue: SUT | red: DIVA adversary",
        f"{label} | step {step} | {status}",
    )
    right = _panel(
        np.asarray(topdown_image),
        "Top-down view | amber: effective Frenet reference",
        "path length={:.1f} m, speed-controller output={:+.2f}".format(
            float(info.get("maneuver_reference_length_m", 0.0)), float(action[3])
        ),
    )
    combined = Image.new("RGB", (left.width + 6 + right.width, height), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + 6, 0))
    return np.asarray(combined)


def _save(frames: list[np.ndarray], output: Path) -> None:
    if not frames:
        raise RuntimeError("replay did not produce render frames")
    images = [Image.fromarray(frame).convert("RGB").quantize(colors=128) for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=70,
        loop=0,
        disposal=2,
        optimize=True,
    )


def run(observations_path: str, design_id: str, output_path: str) -> dict[str, Any]:
    row = _read_row(observations_path, design_id)
    if row.get("schema") != DIVA_SCHEMA:
        raise ValueError("only physically-audited observations may be rendered")
    design_data = row["design"]
    design = DivaCutInDesign(
        int(design_data["candidate_index"]),
        tuple(float(value) for value in design_data["logical_continuous"]),
    )
    task = next(task for task in load_taskbook("mvr/configs/taskbook.json") if task.task_id == row["task_id"])
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        design.scenario_action(),
        episode_seed=int(row["episode_seed"]),
        environment_overrides=VISUAL_ENVIRONMENT_OVERRIDES | {"horizon": 480},
    )
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    behavior = DivaCutInBehavior()

    def capture(current: Any, step: int, info: Mapping[str, Any]) -> None:
        if step == 0:
            current.env.engine.main_camera.track(current.sut)
            _paint_role(current.sut, SUT_COLOR)
            _paint_role(current.adversary, ADVERSARY_COLOR)
            return
        trace.append(
            {
                "step": int(step),
                "adversary_speed_mps": float(current.adversary.speed_km_h) / 3.6,
                "reference_progress": float(info.get("maneuver_reference_progress", 0.0)),
                "longitudinal_request": float(info.get("raw_policy_action", (0.0,) * 4)[3]),
                "maneuver_completed": bool(info.get("semantic_maneuver_completed", False)),
            }
        )
        if step % 4 == 0:
            frames.append(
                _frame(
                    current,
                    info,
                    step,
                    "fixed scenario | red speed {:.2f} m/s".format(
                        behavior.prescribed_speed_mps or 0.0
                    ),
                )
            )

    try:
        rollout = HierarchicalRunner(480).rollout(
            episode,
            "cutin",
            step_action=behavior,
            step_callback=capture,
        )
    finally:
        episode.env.close()
    _save(frames, Path(output_path))
    completion_index = next(
        (index for index, point in enumerate(trace) if point["maneuver_completed"]),
        None,
    )
    through_maneuver = trace if completion_index is None else trace[:completion_index + 1]
    return {
        "schema": "diva_source_replay",
        "source_observations": observations_path,
        "design_id": design_id,
        "task_id": row["task_id"],
        "sut_ref": row["sut_ref"],
        "episode_seed": row["episode_seed"],
        "recorded_status": row["status"],
        "recorded_score": row["score"],
        "recorded_observation_schema": row.get("schema"),
        "replay_design_schema": DIVA_SCHEMA,
        "replay_behavior_contract": BEHAVIOR_CONTRACT,
        "replay_score": valid_critical_score(rollout.outcome),
        "replay_termination_reason": rollout.outcome.get("termination_reason"),
        "frames": len(frames),
        "speed_trace_summary": {
            "minimum_during_maneuver_mps": min(
                point["adversary_speed_mps"] for point in through_maneuver
            ),
            "final_mps_after_episode_continued": trace[-1]["adversary_speed_mps"],
            "maximum_reference_progress": max(point["reference_progress"] for point in trace),
            "maneuver_completed": any(point["maneuver_completed"] for point in trace),
            "maneuver_completion_step": (
                None if completion_index is None else trace[completion_index]["step"]
            ),
        },
        "excluded_from_source_bank": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = run(args.observations, args.design_id, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
