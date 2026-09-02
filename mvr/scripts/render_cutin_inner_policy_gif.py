"""Render stratified Cut-in Inner rollouts with their world-space reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from ..evaluation.support_schedule import FixedQuerySupportSchedule
from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.semantics import quintic_smoothstep
from ..scenario.taskbook import load_taskbook
from ..scenario.task_spec import CUTIN_LOGICAL_PARAMETER_NAMES
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
}
SUT_COLOR = (0.12, 0.43, 0.95)
ADVERSARY_COLOR = (0.92, 0.16, 0.14)


def _representative_actions(task: Any) -> tuple[tuple[str, NormalizedScenarioAction], ...]:
    """Three reproducible points spanning a validation Logical-domain box."""
    bounds = task.logical_domain_bounds
    fractions = {
        "low": (0.75, 0.35, 0.25, 0.25, 0.35, 0.75),
        "medium": (0.50, 0.50, 0.50, 0.50, 0.50, 0.50),
        "high": (0.20, 0.70, 0.70, 0.70, 0.65, 0.25),
    }
    rows = []
    for label, values in fractions.items():
        continuous = tuple(
            float(bounds[name][0] + fraction * (bounds[name][1] - bounds[name][0]))
            for name, fraction in zip(CUTIN_LOGICAL_PARAMETER_NAMES, values)
        )
        for candidate_index in (0, 1):
            rows.append((
                f"{label}_{candidate_index}",
                NormalizedScenarioAction(candidate_index, continuous),
            ))
    return tuple(rows)


def _seed(task: Any, index: int, shots: int, seed: int) -> int:
    source = index if index < shots else 4 + index - shots
    return int(task.geometry_seed + 100_000 * int(seed) + source)


def _label_panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    panel = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    canvas = ImageDraw.Draw(panel)
    canvas.rectangle((0, 0, panel.width, 54), fill=(25, 25, 25))
    canvas.text((12, 8), title, fill=(255, 255, 255))
    canvas.text((12, 30), subtitle, fill=(220, 220, 220))
    return panel


def _dual_view_frame(
    chase_rgb: np.ndarray, topdown_rgb: np.ndarray, policy: str, candidate: str,
    step: int, traffic_status: str,
) -> np.ndarray:
    """Use the repository's historical tail-chase plus global-map GIF layout."""
    height = min(chase_rgb.shape[0], topdown_rgb.shape[0])
    chase = Image.fromarray(chase_rgb).convert("RGB")
    topdown = Image.fromarray(topdown_rgb).convert("RGB")
    chase = chase.resize((round(chase.width * height / chase.height), height))
    topdown = topdown.resize((round(topdown.width * height / topdown.height), height))
    left = _label_panel(
        np.asarray(chase),
        f"SUT tail view | blue IDM SUT | red SAC | {policy}",
        f"candidate: {candidate} | step: {step} | {traffic_status}",
    )
    right = _label_panel(
        np.asarray(topdown),
        "Global top-down | adversary + SUT",
        f"same initial scenario | step: {step}",
    )
    frame = Image.new("RGB", (left.width + 6 + right.width, height), "white")
    frame.paste(left, (0, 0))
    frame.paste(right, (left.width + 6, 0))
    return np.asarray(frame)


def _chase_frame(episode: Any) -> np.ndarray:
    camera = episode.env.engine.get_sensor("main_camera")
    return np.asarray(camera.perceive(to_float=False))[..., :3][:, :, ::-1].copy()


def _paint_role(vehicle: Any, color: tuple[float, float, float]) -> None:
    from panda3d.core import LVecBase4, Material

    vehicle._use_special_color = False
    vehicle._panda_color = color
    material = Material()
    coefficient = float(getattr(vehicle, "MATERIAL_COLOR_COEFF", 1.0))
    material.setBaseColor(LVecBase4(*(min(1.0, channel * coefficient) for channel in color), 1.0))
    material.setMetallic(float(getattr(vehicle, "MATERIAL_METAL_COEFF", 0.0)))
    material.setSpecular(getattr(vehicle, "MATERIAL_SPECULAR_COLOR", (0.0, 0.0, 0.0, 1.0)))
    material.setRoughness(float(getattr(vehicle, "MATERIAL_ROUGHNESS", 0.5)))
    material.setShininess(float(getattr(vehicle, "MATERIAL_SHININESS", 0.0)))
    material.setTwoside(False)
    vehicle.origin.setMaterial(material, True)
    vehicle.origin.setColor(LVecBase4(*color, 1.0), 1000)


def _route_point(route: Any, s_m: float) -> np.ndarray:
    arc = np.asarray(route.arc_lengths_m, dtype=float)
    index = int(np.clip(np.searchsorted(arc, s_m, side="right") - 1, 0, len(arc) - 2))
    fraction = (s_m - arc[index]) / max(arc[index + 1] - arc[index], 1e-6)
    return np.asarray(route.points[index] + fraction * (route.points[index + 1] - route.points[index]), dtype=float)


def _reference_points(episode: Any, samples: int = 81) -> np.ndarray:
    parameters = episode.applied_scenario.logical_parameters
    start = float(parameters["cutin_start_s_m"])
    length = float(parameters["lane_change_length_m"])
    source = episode.adversary_route
    target = episode.sut_route
    points = []
    for q in np.linspace(0.0, 1.0, samples):
        smooth = float(quintic_smoothstep(float(q)))
        s = start + float(q) * length
        points.append((1.0 - smooth) * _route_point(source, s) + smooth * _route_point(target, s))
    return np.asarray(points, dtype=float)


def _paint_reference_path(episode: Any) -> None:
    """Attach the amber quintic reference as dashed geometry in world space."""
    from panda3d.core import LineSegs

    points = _reference_points(episode)
    segments = LineSegs("mvr_cutin_reference")
    segments.setColor(1.0, 0.65, 0.05, 1.0)
    segments.setThickness(3.0)
    for index, (left, right) in enumerate(zip(points[:-1], points[1:])):
        if index % 2:
            continue
        segments.moveTo(float(left[0]), float(left[1]), 0.35)
        segments.drawTo(float(right[0]), float(right[1]), 0.35)
    node = episode.env.engine.render.attachNewNode(segments.create())
    node.setBin("fixed", 50)
    node.setLightOff()
    node.setDepthTest(False)
    node.setDepthWrite(False)


def _draw_reference_overlay(image: np.ndarray, episode: Any, *, view: str) -> np.ndarray:
    """Overlay the same world-space dashed path on both camera products.

    MetaDrive's top-down renderer is a pygame surface independent of Panda3D,
    while the RGB camera uses a tagged display region.  The explicit overlay
    keeps the reference visible in both views even when either renderer omits
    the auxiliary LineSegs node.
    """
    from PIL import ImageDraw

    points = _reference_points(episode)
    projected: list[tuple[float, float]] = []
    if view == "topdown":
        renderer = episode.env.top_down_renderer
        canvas = renderer._frame_canvas
        camera_position = episode.layout.conflict_xy
        pixel_position = canvas.pos2pix(float(camera_position[0]), float(camera_position[1]))
        screen_width, screen_height = renderer._screen_canvas.get_size()
        offset = (pixel_position[0] - screen_width / 2.0, pixel_position[1] - screen_height / 2.0)
        projected = [
            (float(canvas.pos2pix(float(point[0]), float(point[1]))[0] - offset[0]),
             float(canvas.pos2pix(float(point[0]), float(point[1]))[1] - offset[1]))
            for point in points
        ]
    elif view == "chase":
        # ``engine.cam`` is the Panda3D camera NodePath.  The MainCamera
        # sensor's ``camera`` attribute is a ModelNode rig and has no Lens.
        camera = episode.env.engine.cam
        lens = camera.node().getLens()
        from panda3d.core import Point2, Point3

        for point in points:
            relative = camera.getRelativePoint(
                episode.env.engine.render, Point3(float(point[0]), float(point[1]), 0.35)
            )
            ndc = Point2()
            if lens.project(relative, ndc):
                projected.append((
                    (float(ndc.x) + 1.0) * 0.5 * image.shape[1],
                    (1.0 - float(ndc.y)) * 0.5 * image.shape[0],
                ))
    else:
        raise ValueError(f"unknown reference overlay view: {view}")

    canvas_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas_image)
    color = (255, 166, 15)
    for index, (left, right) in enumerate(zip(projected[:-1], projected[1:])):
        if index % 2:
            continue
        draw.line((left, right), fill=color, width=3)
    return np.asarray(canvas_image)


def _capture_query_frames(
    shots: int, label: str, every: int = 2,
) -> tuple[list[np.ndarray], Callable[..., None]]:
    frames: list[np.ndarray] = []
    episode_order: dict[int, int] = {}

    def capture(episode: Any, step: int, info: Mapping[str, Any]) -> None:
        episode_number = episode_order.setdefault(id(episode), len(episode_order))
        if episode_number != shots:
            return
        camera = episode.env.engine.main_camera
        if step == 0:
            camera.track(episode.sut)
            _paint_role(episode.sut, SUT_COLOR)
            _paint_role(episode.adversary, ADVERSARY_COLOR)
            _paint_reference_path(episode)
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
        chase = _draw_reference_overlay(_chase_frame(episode), episode, view="chase")
        topdown = _draw_reference_overlay(np.asarray(topdown), episode, view="topdown")
        if info.get("traffic_shield_rejected"):
            traffic_status = str(info.get("traffic_shield_rejection_reason"))
        elif info.get("semantic_maneuver_completed"):
            traffic_status = "maneuver complete"
        elif info.get("semantic_maneuver_active"):
            traffic_status = "maneuver active"
        else:
            traffic_status = "lane following"
        frames.append(_dual_view_frame(
            chase, np.asarray(topdown), label, episode.layout.candidate, step, traffic_status,
        ))

    return frames, capture


def _save_gif(frames: list[np.ndarray], output: Path) -> None:
    if not frames:
        raise RuntimeError("policy rollout did not yield any render frames")
    images = [Image.fromarray(frame).convert("RGB").quantize(colors=128) for frame in frames]
    images[0].save(
        output, save_all=True, append_images=images[1:], duration=50, loop=0, disposal=2, optimize=True,
    )


def run(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    geometry_id: str | None = None,
) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    cutin_inner = config.get("cutin_inner")
    if cutin_inner is None or bool(cutin_inner.get("allow_outer", True)):
        raise ValueError("GIF rendering requires the no-Outer Cut-in Inner configuration")
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    allowed_stages = {
        TrainingStage.INTERACTION_PRIOR.value,
        TrainingStage.CONTEXT_META.value,
    }
    if checkpoint.stage not in allowed_stages:
        raise ValueError("GIF rendering requires an Inner SAC checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    geometry_ids = (geometry_id,) if geometry_id else cutin_inner.get("validation_geometry_ids", ())
    tasks = select_cutin_validation_tasks(load_taskbook(taskbook_path), geometry_ids)
    seed = int(config["evaluation"]["seeds"][0])
    criteria = FailureCriteria.from_config(config["failure"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports = []
    variants = (
        ((0, "interaction prior (K=0)", "cutin_interaction_prior_k0.gif"),)
        if checkpoint.stage == TrainingStage.INTERACTION_PRIOR.value
        else (
            (0, "shared prior (K=0)", "cutin_shared_prior_k0.gif"),
            (4, "adapted h+z (K=4)", "cutin_adapted_k4.gif"),
        )
    )
    for task in tasks:
        representatives = _representative_actions(task)
        support_provider = FixedQuerySupportSchedule(
            task, representatives[0][1], 4, 4, seed,
        )
        support_online = build_online(
            model, task, int(config["training"]["step_budget"]), criteria,
        )
        support_episodes = support_online.run(
            task, 4, deterministic=True, posterior_support_limit=4,
            scene_action_provider=support_provider,
            episode_seed_provider=lambda current, index: _seed(current, index, 4, seed),
        ).episodes
        posterior_z4 = support_episodes[-1].latent_after
        support_queries_k4 = support_provider.provenance()
        for risk_label, query_action in representatives:
            for shots, label, filename in variants:
                frames, callback = _capture_query_frames(0, label)
                online = build_online(
                    model, task, int(config["training"]["step_budget"]), criteria
                )
                episodes = online.run(
                    task,
                    1, deterministic=True, posterior_support_limit=0,
                    scene_action_provider=lambda _task, _index, _candidates, _space: query_action,
                    episode_seed_provider=lambda current, index: _seed(
                        current, index, 0, seed,
                    ),
                    initial_latent=posterior_z4 if shots == 4 else None,
                    rollout_step_callback=callback,
                    environment_overrides={
                        **VISUAL_ENVIRONMENT_OVERRIDES,
                        "camera_height": 3.2,
                        "camera_dist": 8.0,
                        "camera_pitch": 12.0,
                        "camera_smooth": False,
                    },
                ).episodes
                query = episodes[0]
                support_queries = support_queries_k4 if shots == 4 else ()
                gif_path = output_path / (
                    f"{task.geometry_id}_{task.sut_ref}_{risk_label}_{filename}"
                )
                _save_gif(frames, gif_path)
                post_onset = next(
                    (
                        row for row in query.rollout.transitions
                        if bool(row["info"].get("semantic_maneuver_started", False))
                    ),
                    query.rollout.transitions[0],
                )
                logical_parameters = {
                    name: float(mvr_parameter_spaces()["cutin"].decode(query_action)[name])
                    for name in CUTIN_LOGICAL_PARAMETER_NAMES
                }
                logical_parameters["cutin_start_s_m"] = float(
                    query.rollout.transitions[0]["info"]["cutin_reference_start_s_m"]
                )
                reports.append({
                    "policy": label,
                    "support_shots": shots,
                    "gif": str(gif_path),
                    "task_id": task.task_id,
                    "geometry_id": task.geometry_id,
                    "risk_stratum": risk_label.rsplit("_", 1)[0],
                    "logical_parameters": logical_parameters,
                    "support_queries": support_queries,
                    "frames": len(frames),
                    "outcome": dict(query.outcome),
                    "first_post_onset_policy_action": [
                        float(value)
                        for value in np.asarray(post_onset["raw_action"], dtype=float)
                    ],
                    "first_post_onset_executed_action": [
                        float(value) for value in np.asarray(
                            post_onset["executed_action"], dtype=float,
                        )
                    ],
                    "first_post_onset_target_lateral_m": float(
                        post_onset["info"].get("traffic_cutin_lateral_m", 0.0)
                    ),
                    "cutin_completed": any(
                        bool(row["info"].get("semantic_maneuver_completed", False))
                        for row in query.rollout.transitions
                    ),
                    "valid_event": bool(
                        query.outcome["valid_target_collision"]
                        or query.outcome["valid_critical_near_miss"]
                    ),
                })
    report = {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "geometry_split": "train",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "representative_queries": "low/medium/high × left/right candidate for every selected validation task",
        "support_protocol": "K=4 infers z from one deterministic task-local support set distinct from the representative queries; policy weights are frozen.",
        "rollouts": reports,
    }
    (output_path / "cutin_policy_gif_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geometry-id")
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, args.geometry_id)


if __name__ == "__main__":
    main()
