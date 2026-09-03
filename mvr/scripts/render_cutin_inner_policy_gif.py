"""Render validation-selected Cut-in rollouts in the historical dual view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from ..evaluation.cutin_query_design import build_cutin_validation_queries
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


VISUAL_ENVIRONMENT_OVERRIDES = {
    "image_observation": True,
    "window_size": (640, 360),
    "interface_panel": [],
    "show_interface": False,
    "show_logo": False,
    "show_fps": False,
    "sensors": {"main_camera": ()},
    "vehicle_config": {"image_source": "main_camera"},
    "camera_height": 3.2,
    "camera_dist": 8.0,
    "camera_pitch": 12.0,
    "camera_smooth": False,
}
SUT_COLOR = (0.12, 0.43, 0.95)
ADVERSARY_COLOR = (0.92, 0.16, 0.14)


def _paint_role(vehicle: Any, color: tuple[float, float, float]) -> None:
    from panda3d.core import LVecBase4, Material

    vehicle._use_special_color = False
    vehicle._panda_color = color
    material = Material()
    coefficient = float(getattr(vehicle, "MATERIAL_COLOR_COEFF", 1.0))
    material.setBaseColor(LVecBase4(*(
        min(1.0, channel * coefficient) for channel in color
    ), 1.0))
    material.setMetallic(float(getattr(vehicle, "MATERIAL_METAL_COEFF", 0.0)))
    material.setSpecular(
        getattr(vehicle, "MATERIAL_SPECULAR_COLOR", (0.0, 0.0, 0.0, 1.0))
    )
    material.setRoughness(float(getattr(vehicle, "MATERIAL_ROUGHNESS", 0.5)))
    material.setShininess(float(getattr(vehicle, "MATERIAL_SHININESS", 0.0)))
    material.setTwoside(False)
    vehicle.origin.setMaterial(material, True)
    vehicle.origin.setColor(LVecBase4(*color, 1.0), 1000)


def _chase_frame(episode: Any) -> np.ndarray:
    camera = episode.env.engine.get_sensor("main_camera")
    return np.asarray(camera.perceive(to_float=False))[..., :3][:, :, ::-1].copy()


def _label_panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    panel = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    canvas = ImageDraw.Draw(panel)
    canvas.rectangle((0, 0, panel.width, 58), fill=(25, 25, 25))
    canvas.text((12, 8), title, fill=(255, 255, 255))
    canvas.text((12, 32), subtitle, fill=(220, 220, 220))
    return panel


def _dual_view_frame(
    chase: np.ndarray,
    topdown: np.ndarray,
    policy: str,
    candidate: str,
    step: int,
    status: str,
    planner_action: list[float],
) -> np.ndarray:
    """Preserve the repository's tail-chase plus global-map GIF layout."""
    height = min(chase.shape[0], topdown.shape[0])
    left_source = Image.fromarray(chase).convert("RGB")
    right_source = Image.fromarray(topdown).convert("RGB")
    left_source = left_source.resize((round(left_source.width * height / left_source.height), height))
    right_source = right_source.resize((round(right_source.width * height / right_source.height), height))
    action_text = "L={:+.2f} b1={:+.2f} b2={:+.2f} lon={:+.2f}".format(*planner_action)
    left = _label_panel(
        np.asarray(left_source),
        f"SUT tail view | blue IDM SUT | red SAC | {policy}",
        f"{candidate} | step {step} | {status} | {action_text}",
    )
    right = _label_panel(
        np.asarray(right_source),
        "Global top-down | amber dashed current Frenet path",
        f"same validation query | step {step}",
    )
    frame = Image.new("RGB", (left.width + 6 + right.width, height), "white")
    frame.paste(left, (0, 0))
    frame.paste(right, (left.width + 6, 0))
    return np.asarray(frame)


def _draw_reference_overlay(
    image: np.ndarray,
    episode: Any,
    points_xy: Any,
    *,
    view: str,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    projected: list[tuple[float, float]] = []
    if view == "topdown":
        renderer = episode.env.top_down_renderer
        canvas = renderer._frame_canvas
        camera_position = episode.layout.conflict_xy
        pixel = canvas.pos2pix(float(camera_position[0]), float(camera_position[1]))
        width, height = renderer._screen_canvas.get_size()
        offset = (pixel[0] - width / 2.0, pixel[1] - height / 2.0)
        projected = [
            (
                float(canvas.pos2pix(float(point[0]), float(point[1]))[0] - offset[0]),
                float(canvas.pos2pix(float(point[0]), float(point[1]))[1] - offset[1]),
            )
            for point in points
        ]
    elif view == "chase":
        from panda3d.core import Point2, Point3

        camera = episode.env.engine.cam
        lens = camera.node().getLens()
        for point in points:
            relative = camera.getRelativePoint(
                episode.env.engine.render,
                Point3(float(point[0]), float(point[1]), 0.35),
            )
            ndc = Point2()
            if lens.project(relative, ndc):
                projected.append((
                    (float(ndc.x) + 1.0) * 0.5 * image.shape[1],
                    (1.0 - float(ndc.y)) * 0.5 * image.shape[0],
                ))
    else:
        raise ValueError(f"unknown reference overlay view: {view}")
    rendered = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for index, (left, right) in enumerate(zip(projected[:-1], projected[1:])):
        if index % 2 == 0:
            draw.line((left, right), fill=(255, 166, 15), width=3)
    return np.asarray(rendered)


def _capture_frames(
    label: str, every: int = 2
) -> tuple[list[np.ndarray], Callable[..., None]]:
    frames: list[np.ndarray] = []

    def capture(episode: Any, step: int, info: Mapping[str, Any]) -> None:
        if step == 0:
            episode.env.engine.main_camera.track(episode.sut)
            _paint_role(episode.sut, SUT_COLOR)
            _paint_role(episode.adversary, ADVERSARY_COLOR)
            return
        if step % every:
            return
        topdown = episode.env.render(
            mode="topdown",
            window=False,
            screen_size=(800, 800),
            scaling=5,
            camera_position=episode.layout.conflict_xy,
        )
        points = info["maneuver_reference_points_xy"]
        chase = _draw_reference_overlay(
            _chase_frame(episode), episode, points, view="chase"
        )
        topdown = _draw_reference_overlay(
            np.asarray(topdown), episode, points, view="topdown"
        )
        if info.get("traffic_shield_rejected"):
            status = str(info.get("traffic_shield_rejection_reason"))
        elif info.get("semantic_maneuver_completed"):
            status = "maneuver complete"
        elif info.get("semantic_maneuver_active"):
            status = "maneuver active"
        else:
            status = "lane following"
        frames.append(_dual_view_frame(
            chase,
            topdown,
            label,
            episode.layout.candidate,
            step,
            status,
            [float(value) for value in info.get("planner_action", ())],
        ))

    return frames, capture


def _save_gif(frames: list[np.ndarray], output: Path) -> None:
    if not frames:
        raise RuntimeError("policy rollout did not yield render frames")
    images = [
        Image.fromarray(frame).convert("RGB").quantize(colors=128)
        for frame in frames
    ]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=50,
        loop=0,
        disposal=2,
        optimize=True,
    )


def _support_provider(rows: list[Mapping[str, Any]]) -> Callable[..., NormalizedScenarioAction]:
    actions = [
        NormalizedScenarioAction(
            int(row["candidate_index"]),
            tuple(float(value) for value in row["continuous"]),
        )
        for row in rows
    ]

    def provider(
        _task: Any, index: int, _candidates: Any, _space: Any
    ) -> NormalizedScenarioAction:
        return actions[index]

    return provider


def run(
    config_path: str,
    checkpoint_path: str,
    validation_path: str,
    output_dir: str,
) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    if checkpoint.stage != TrainingStage.CONTEXT_META.value:
        raise ValueError("GIF rendering requires a context_meta checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    validation = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    if validation["scope"]["test_split_accessed"]:
        raise ValueError("GIF rendering refuses a report that accessed test split")
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    tasks = {
        task.task_id: task
        for task in select_cutin_validation_tasks(load_taskbook(taskbook_path))
    }
    seed = int(config["evaluation"]["seeds"][0])
    criteria = FailureCriteria.from_config(config["failure"])
    design = config["evaluation"]["query_design"]
    candidates = len(mvr_parameter_spaces()["cutin"].candidates)
    support_rows = {
        row["task_id"]: row["nested_support"]
        for row in validation["support_provenance"]
        if int(row["seed"]) == seed
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for selected in validation["gif_selection"]:
        task = tasks[selected["task_id"]]
        query_rows = build_cutin_validation_queries(
            task,
            candidates=candidates,
            sobol_interior=int(design["sobol_interior_per_candidate"]),
            boundary=int(design["boundary_per_candidate"]),
            seed=int(config["evaluation"]["paired_bootstrap_seed"]),
        )
        query_index, query = next(
            (index, row) for index, row in enumerate(query_rows)
            if row.query_id == selected["query_id"]
        )
        online = build_online(
            model, task, int(config["training"]["step_budget"]), criteria
        )
        support = support_rows[task.task_id]
        support_result = online.run(
            task,
            4,
            deterministic=True,
            posterior_support_limit=4,
            scene_action_provider=_support_provider(support),
            episode_seed_provider=lambda current, index: int(
                current.geometry_seed + 10_000 * seed + index
            ),
        )
        z4 = support_result.episodes[-1].latent_after
        for shots, label in ((0, "shared prior K=0"), (4, "adapted K=4")):
            frames, callback = _capture_frames(label)
            episode_seed = int(task.geometry_seed + 100_000 * seed + query_index)
            episode = online.run(
                task,
                1,
                deterministic=True,
                posterior_support_limit=0,
                initial_latent=z4 if shots == 4 else None,
                scene_action_provider=(
                    lambda _task, _index, _candidates, _space, action=query.action: action
                ),
                episode_seed_provider=lambda _task, _index, value=episode_seed: value,
                rollout_step_callback=callback,
                environment_overrides=VISUAL_ENVIRONMENT_OVERRIDES,
            ).episodes[0]
            filename = (
                f"c{selected['candidate_index']}_{selected['risk_stratum']}_k{shots}.gif"
            )
            path = output / filename
            _save_gif(frames, path)
            post_onset = next(
                (
                    row for row in episode.rollout.transitions
                    if row["info"].get("semantic_maneuver_started", False)
                ),
                episode.rollout.transitions[0],
            )
            report_rows.append({
                "gif": str(path),
                "task_id": task.task_id,
                "query_id": query.query_id,
                "candidate_index": selected["candidate_index"],
                "risk_stratum": selected["risk_stratum"],
                "support_shots": shots,
                "seed": seed,
                "normalized_parameters": list(query.action.continuous),
                "logical_parameters": dict(episode.concrete_scenario.logical_parameters),
                "frames": len(frames),
                "first_post_onset_raw_policy_action": [
                    float(value) for value in post_onset["raw_policy_action"]
                ],
                "first_post_onset_planner_action": [
                    float(value) for value in post_onset["planner_action"]
                ],
                "first_post_onset_executed_vehicle_action": [
                    float(value) for value in post_onset["executed_vehicle_action"]
                ],
                "cutin_completed": any(
                    row["info"].get("semantic_maneuver_completed", False)
                    for row in episode.rollout.transitions
                ),
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
        "selection": "actual validation K=0 risk 10/50/90 percentile per candidate",
        "style": "historical tail-chase plus global top-down dual view",
        "reference": "dynamic amber dashed effective Frenet path",
        "rollouts": report_rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.validation, args.output_dir)


if __name__ == "__main__":
    main()
