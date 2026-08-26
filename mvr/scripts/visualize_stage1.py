"""Create auditable top-down Stage 1 replays for the validated MVR checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..failure.criteria import FailureCriteria
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.concrete import ConcreteScenario
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
)
from ..training.stage1_sampling import PretrainSceneSampler
from ..training.stages import TrainingStage
from ..training.trainers import build_online


@dataclass(frozen=True)
class CapturedReplay:
    policy: str
    scenario: ConcreteScenario
    outcome: Mapping[str, Any]
    frames: tuple[np.ndarray, ...]
    closest_frame: int


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


def _rank(outcome: Mapping[str, Any]) -> tuple[float, ...]:
    """Prefer valid critical outcomes, then the closest, fastest encounter."""
    valid = bool(outcome.get("is_valid_episode", True))
    valid_collision = bool(
        outcome.get("valid_target_collision", outcome.get("target_collision", False) and valid)
    ) and valid
    return (
        float(valid),
        float(outcome.get("is_failure", False)),
        float(valid_collision),
        float(outcome.get("valid_critical_near_miss", False)),
        -float(outcome["min_ttc"]),
        float(outcome["max_closing_speed"]),
        -float(outcome["min_distance"]),
    )


def select_representative(episodes: Sequence[Any]) -> Any:
    if not episodes:
        raise ValueError("cannot select a representative from no episodes")
    return max(episodes, key=lambda episode: _rank(episode.outcome))


def _closest_frame(transitions: Sequence[Mapping[str, Any]]) -> int:
    if not transitions:
        return 0
    return min(
        range(len(transitions)),
        key=lambda index: float(transitions[index]["trajectory_features"][10]),
    )


def _policy_provider(name: str) -> Callable[[np.ndarray], np.ndarray] | None:
    if name == "zero":
        return lambda _: np.zeros(2, dtype=np.float32)
    if name == "trained_inner":
        return None
    raise ValueError(f"unsupported visualization policy {name!r}")


def _write_gif(frames: Sequence[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("cannot write an empty replay")
    images = [
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB").quantize(colors=128)
        for frame in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        optimize=True,
    )


def _label_panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    panel = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    canvas = ImageDraw.Draw(panel)
    canvas.rectangle((0, 0, panel.width, 54), fill=(25, 25, 25))
    canvas.text((12, 8), title, fill=(255, 255, 255))
    canvas.text((12, 30), subtitle, fill=(220, 220, 220))
    return panel


def _dual_view_frame(
    chase_rgb: np.ndarray,
    topdown_rgb: np.ndarray,
    policy: str,
    candidate: str,
    step: int,
    traffic_status: str,
) -> np.ndarray:
    """Reuse the historical SUT-following plus global-map presentation."""
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

    # Some MetaDrive vehicle presets replace panda_color with a palette entry.
    # Disable that presentation-only override before applying the role material.
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


def _capture(
    online: Any,
    task: Any,
    scenario: ConcreteScenario,
    episode_index: int,
    policy: str,
    topdown: Mapping[str, Any],
    tail_camera: Mapping[str, Any],
) -> CapturedReplay:
    frames: list[np.ndarray] = []
    stride = int(topdown["frame_stride"])
    screen_size = tuple(int(value) for value in topdown["screen_size"])

    def callback(episode: Any, step: int, info: Mapping[str, Any]) -> None:
        camera = episode.env.engine.main_camera
        if step == 0:
            camera.track(episode.sut)
            _paint_role(episode.sut, SUT_COLOR)
            _paint_role(episode.adversary, ADVERSARY_COLOR)
            return
        if step % stride:
            return
        topdown_frame = episode.env.render(
            mode="topdown",
            window=False,
            screen_size=screen_size,
            scaling=int(topdown["scaling"]),
            camera_position=episode.layout.conflict_xy,
        )
        if info.get("traffic_shield_rejected"):
            traffic_status = str(info.get("traffic_shield_rejection_reason"))
        elif info.get("traffic_lane_change_completed"):
            traffic_status = "legal merge complete"
        elif info.get("traffic_lane_change_started"):
            traffic_status = "legal merge active"
        else:
            traffic_status = "lane following"
        route_progress = episode.adversary_route.projection(
            episode.adversary.position,
            episode.adversary.heading_theta,
        ).s_m
        traffic_status = f"{traffic_status} | route: {route_progress:.0f}/{episode.adversary_route.length_m:.0f} m"
        frames.append(_dual_view_frame(
            _chase_frame(episode),
            np.asarray(topdown_frame),
            policy,
            scenario.candidate_id,
            step,
            traffic_status,
        ))

    action = scenario.replay_action(mvr_parameter_spaces()[task.functional_scenario])
    environment_overrides = {
        **VISUAL_ENVIRONMENT_OVERRIDES,
        "camera_height": float(tail_camera["height"]),
        "camera_dist": float(tail_camera["distance"]),
        "camera_pitch": float(tail_camera["pitch"]),
        "camera_smooth": False,
    }
    result = online.run(
        task,
        1,
        deterministic=True,
        posterior_support_limit=0,
        episode_index_offset=episode_index,
        scene_action_provider=lambda *_: action,
        inner_action_provider=_policy_provider(policy),
        rollout_step_callback=callback,
        environment_overrides=environment_overrides,
    )
    episode = result.episodes[0]
    if not frames:
        raise RuntimeError("top-down renderer produced no frames")
    return CapturedReplay(
        policy,
        episode.concrete_scenario,
        episode.outcome,
        tuple(frames),
        _closest_frame(episode.rollout.transitions) // stride,
    )


def run(config_path: str | Path, checkpoint_path: str | Path, output: str | Path) -> dict[str, Any]:
    from panda3d.core import loadPrcFileData

    with tempfile.TemporaryDirectory(prefix="mvr_panda3d_") as cache_dir:
        loadPrcFileData("", f"model-cache-dir {Path(cache_dir).as_posix()}")
        return _run(config_path, checkpoint_path, output)


def _run(config_path: str | Path, checkpoint_path: str | Path, output: str | Path) -> dict[str, Any]:
    config, taskbook_path, device = load_config(config_path)
    settings = dict(config["visualization"])
    topdown = dict(settings["topdown"])
    tail_camera = dict(settings["tail_camera"])
    policies = tuple(str(value) for value in settings["policies"])
    if "trained_inner" not in policies:
        raise ValueError("visualization.policies must include trained_inner for case selection")
    if min(int(topdown["frame_stride"]), int(topdown["fps"]), int(topdown["scaling"])) < 1:
        raise ValueError("top-down scaling, frame_stride, and fps must be positive")
    if min(float(tail_camera["height"]), float(tail_camera["distance"])) <= 0:
        raise ValueError("tail-camera height and distance must be positive")

    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config)
    )
    if checkpoint.stage != TrainingStage.INNER_PRETRAIN.value:
        raise ValueError("Stage 1 visualization requires an inner_pretrain checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook_path)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    criteria = FailureCriteria.from_config(config["failure"])
    tasks = [
        task for task in load_taskbook(taskbook_path)
        if task.sut_split == "validation"
        and task.geometry_split == "validation"
        and task.functional_split == "train"
    ]
    if not tasks:
        raise ValueError("taskbook has no Stage 1 validation tasks")
    sampler = PretrainSceneSampler(
        tuple(tasks), int(settings["cases_per_task"]), int(config["seed"])
    )

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for task in tasks:
        online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
        candidates = online.run(
            task,
            int(settings["cases_per_task"]),
            deterministic=True,
            posterior_support_limit=0,
            scene_action_provider=sampler,
        ).episodes
        selected = select_representative(candidates)
        selected_index = int(selected.episode_id.rsplit(":", maxsplit=1)[1])
        family_dir = destination / task.functional_scenario
        family_dir.mkdir(exist_ok=True)
        replays = [
            _capture(
                online,
                task,
                selected.concrete_scenario,
                selected_index,
                policy,
                topdown,
                tail_camera,
            )
            for policy in policies
        ]
        for replay in replays:
            _write_gif(replay.frames, family_dir / f"{replay.policy}.gif", int(topdown["fps"]))
        reports.append({
            "family": task.functional_scenario,
            "task_id": task.task_id,
            "selected_episode_index": selected_index,
            "selection_outcome": dict(selected.outcome),
            "selected_scenario": selected.concrete_scenario.to_dict(),
            "replays": [
                {
                    "policy": replay.policy,
                    "scenario": replay.scenario.to_dict(),
                    "outcome": dict(replay.outcome),
                    "frames": len(replay.frames),
                    "closest_frame": replay.closest_frame,
                }
                for replay in replays
            ],
        })
    report = {
        "mode": "stage1_dual_view_gif_visualization",
        "renderer": "metadrive_offscreen_tail_chase_topdown",
        "media": "gif",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "policies": list(policies),
        "families": reports,
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
