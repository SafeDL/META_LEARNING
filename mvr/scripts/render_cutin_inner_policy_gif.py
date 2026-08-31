"""Render actual fixed-x0 Cut-in Inner policy rollouts as off-screen GIFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
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
}
SUT_COLOR = (0.12, 0.43, 0.95)
ADVERSARY_COLOR = (0.92, 0.16, 0.14)


def _x0(config: Mapping[str, Any]) -> NormalizedScenarioAction:
    payload = config["evaluation"]["fixed_query_x0"]
    return NormalizedScenarioAction(
        int(payload["candidate_index"]),
        tuple(float(value) for value in payload["continuous"]),
    )


def _provider(action: NormalizedScenarioAction):
    return lambda _task, _index, _candidates, _space: action


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
        if info.get("traffic_shield_rejected"):
            traffic_status = str(info.get("traffic_shield_rejection_reason"))
        elif info.get("semantic_maneuver_completed"):
            traffic_status = "maneuver complete"
        elif info.get("semantic_maneuver_active"):
            traffic_status = "maneuver active"
        else:
            traffic_status = "lane following"
        frames.append(_dual_view_frame(
            _chase_frame(episode), np.asarray(topdown), label, episode.layout.candidate, step, traffic_status,
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
    if checkpoint.stage != TrainingStage.CONTEXT_META.value:
        raise ValueError("GIF rendering requires a context_meta checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    geometry_ids = (geometry_id,) if geometry_id else cutin_inner.get("validation_geometry_ids", ())
    tasks = select_cutin_validation_tasks(load_taskbook(taskbook_path), geometry_ids)
    if len(tasks) != 1:
        raise ValueError("GIF configuration must select exactly one validation task")
    task = tasks[0]
    x0 = _x0(config)
    seed = int(config["evaluation"]["seeds"][0])
    criteria = FailureCriteria.from_config(config["failure"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports = []
    for shots, label, filename in (
        (0, "shared prior (K=0)", "cutin_shared_prior_k0.gif"),
        (4, "adapted h+z (K=4)", "cutin_adapted_k4.gif"),
    ):
        frames, callback = _capture_query_frames(shots, label)
        online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
        episodes = online.run(
            task,
            shots + 1,
            deterministic=True,
            posterior_support_limit=shots,
            scene_action_provider=_provider(x0),
            episode_seed_provider=lambda current, index, value=shots: _seed(
                current, index, value, seed,
            ),
            rollout_step_callback=callback,
            environment_overrides={
                **VISUAL_ENVIRONMENT_OVERRIDES,
                "camera_height": 3.2,
                "camera_dist": 8.0,
                "camera_pitch": 12.0,
                "camera_smooth": False,
            },
        ).episodes
        query = episodes[-1]
        gif_path = output_path / filename
        _save_gif(frames, gif_path)
        reports.append({
            "policy": label,
            "support_shots": shots,
            "gif": str(gif_path),
            "frames": len(frames),
            "outcome": dict(query.outcome),
            "first_inner_action": [
                float(value)
                for value in np.asarray(
                    query.rollout.transitions[0].get(
                        "executed_action", query.rollout.transitions[0]["action"]
                    ),
                    dtype=float,
                )
            ],
        })
    report = {
        "scope": {
            "functional_scenario": "cutin",
            "sut_split": "validation",
            "logical_split": "validation",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "fixed_query_x0": {"candidate_index": x0.candidate_index, "continuous": list(x0.continuous)},
        "task_id": task.task_id,
        "geometry_id": task.geometry_id,
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
