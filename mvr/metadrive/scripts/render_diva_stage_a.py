"""Render Stage A source evidence and one exact hazardous Cut-in replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import yaml

from ..diva.behavior import DivaCutInBehavior
from ..diva.source_bank import retained_task, study_task
from ..diva.types import DIVA_SCHEMA, DivaCutInDesign
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner


SOURCE_ORDER = (
    "idm_cautious",
    "idm_defensive",
    "idm_normal",
    "idm_assertive",
)
SOURCE_LABELS = ("Cautious", "Defensive", "Normal", "Assertive")
SOURCE_COLORS = ("#4C78A8", "#72B7B2", "#F2CF5B", "#E45756")
VISUAL_OVERRIDES = {
    "image_observation": True,
    "window_size": (640, 360),
    "interface_panel": [],
    "show_interface": False,
    "show_logo": False,
    "show_fps": False,
    "sensors": {"main_camera": ()},
    "vehicle_config": {"image_source": "main_camera"},
    "camera_smooth": False,
}


def _load_observations(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(row.get("schema") != DIVA_SCHEMA for row in rows):
        raise ValueError("Stage A visualization requires v2 source observations")
    return rows


def _plot_summary(
    rows: list[Mapping[str, Any]],
    loso: Mapping[str, Any],
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle("DIVA-Mine Stage A | Cut-in source evidence", fontsize=18, fontweight="bold")

    event_counts = []
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row["sut_ref"] == source]
        event_counts.append((
            sum(float(row["score"]) == 0.0 for row in source_rows),
            sum(float(row["score"]) == 0.5 for row in source_rows),
            sum(float(row["score"]) == 1.0 for row in source_rows),
        ))
    safe, near_miss, collision = (np.asarray(values) for values in zip(*event_counts))
    positions = np.arange(len(SOURCE_ORDER))
    axis = axes[0, 0]
    axis.bar(positions, safe, label="No critical event", color="#D9E2EC")
    axis.bar(positions, near_miss, bottom=safe, label="Critical near-miss", color="#F2CF5B")
    axis.bar(positions, collision, bottom=safe + near_miss, label="Collision", color="#E45756")
    for index, value in enumerate(collision + near_miss):
        axis.text(index, 65, f"{value} event", ha="center", va="bottom", fontweight="bold")
    axis.set_title("Formal outcome of 64 shared scenarios per SUT")
    axis.set_ylabel("Simulator calls")
    axis.set_xticks(positions, SOURCE_LABELS)
    axis.set_ylim(0, 72)
    axis.legend(loc="upper left", frameon=True)

    axis = axes[0, 1]
    rng = np.random.default_rng(11)
    for index, (source, color) in enumerate(zip(SOURCE_ORDER, SOURCE_COLORS)):
        values = [row["vulnerability_response"] for row in rows if row["sut_ref"] == source]
        jitter = rng.uniform(-0.13, 0.13, len(values))
        axis.scatter(np.full(len(values), index) + jitter, values, s=28, alpha=0.75, color=color)
        axis.hlines(np.mean(values), index - 0.28, index + 0.28, color="#202020", linewidth=2.2)
    axis.axhspan(0.75, 1.0, color="#FDE2E1", zorder=0, label="Event-response band")
    axis.set_title("Continuous vulnerability response")
    axis.set_ylabel("Response used for prior fitting")
    axis.set_xticks(positions, SOURCE_LABELS)
    axis.set_ylim(0, 1.05)
    axis.legend(loc="upper left", frameon=True)

    by_design: dict[str, dict[str, float]] = {}
    for row in rows:
        by_design.setdefault(row["design"]["design_id"], {})[row["sut_ref"]] = row["vulnerability_response"]
    designs = sorted(by_design, key=lambda design: np.mean(list(by_design[design].values())), reverse=True)
    response_matrix = np.asarray([
        [by_design[design][source] for design in designs] for source in SOURCE_ORDER
    ])
    axis = axes[1, 0]
    image = axis.imshow(response_matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title("Same 64 scenarios, different controller responses")
    axis.set_xlabel("Shared designs, ordered from high to low mean response")
    axis.set_ylabel("Source SUT")
    axis.set_yticks(np.arange(len(SOURCE_ORDER)), SOURCE_LABELS)
    axis.set_xticks([])
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
    colorbar.set_label("Vulnerability response")

    axis = axes[1, 1]
    aggregates = loso["aggregates"]
    methods = (
        ("shared_prior_k0", "Shared prior", "#4C78A8", "o"),
        ("random_support_k", "Random support", "#9D9DA3", "s"),
        ("diagnostic_support_k", "DIVA diagnostic", "#E45756", "D"),
    )
    for prefix, label, color, marker in methods:
        if prefix == "shared_prior_k0":
            axis.scatter([0], [aggregates[prefix]["ndcg_at_8"]], s=90, color=color, marker=marker, label=label, zorder=4)
            continue
        shots = np.asarray((1, 2, 4))
        values = np.asarray([aggregates[f"{prefix}{shot}"]["ndcg_at_8"] for shot in shots])
        axis.plot(shots, values, color=color, linewidth=2.5, marker=marker, markersize=7, label=label)
    axis.set_title("Leave-one-source-out top-8 ranking quality")
    axis.set_xlabel("Diagnostic scenarios observed (K)")
    axis.set_ylabel("NDCG@8")
    axis.set_xticks((0, 1, 2, 4))
    axis.set_ylim(0.80, 1.0)
    axis.legend(loc="lower right", frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def _paint_role(vehicle: Any, color: tuple[float, float, float]) -> None:
    from panda3d.core import LVecBase4, Material

    vehicle._use_special_color = False
    vehicle._panda_color = color
    material = Material()
    material.setBaseColor(LVecBase4(*color, 1.0))
    vehicle.origin.setMaterial(material, True)
    vehicle.origin.setColor(LVecBase4(*color, 1.0), 1000)


def _chase_frame(episode: Any) -> np.ndarray:
    camera = episode.env.engine.get_sensor("main_camera")
    return np.asarray(camera.perceive(to_float=False))[..., :3][:, :, ::-1].copy()


def _draw_reference_overlay(
    image: np.ndarray,
    episode: Any,
    points_xy: Any,
    *,
    view: str,
    camera_position: tuple[float, float] | None = None,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    projected: list[tuple[float, float]] = []
    if view == "topdown":
        renderer = episode.env.top_down_renderer
        canvas = renderer._frame_canvas
        view_position = camera_position or episode.layout.conflict_xy
        pixel = canvas.pos2pix(float(view_position[0]), float(view_position[1]))
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
                episode.env.engine.render, Point3(float(point[0]), float(point[1]), 0.35)
            )
            ndc = Point2()
            if lens.project(relative, ndc):
                projected.append((
                    (float(ndc.x) + 1.0) * 0.5 * image.shape[1],
                    (1.0 - float(ndc.y)) * 0.5 * image.shape[0],
                ))
    else:
        raise ValueError(f"unknown replay view: {view}")
    rendered = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for index, (left, right) in enumerate(zip(projected[:-1], projected[1:])):
        if index % 2 == 0:
            draw.line((left, right), fill=(255, 166, 15), width=3)
    return np.asarray(rendered)


def _panel(image: np.ndarray, title: str, subtitle: str) -> Image.Image:
    panel = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 50), fill=(25, 25, 25))
    draw.text((10, 7), title, fill=(255, 255, 255))
    draw.text((10, 28), subtitle, fill=(220, 220, 220))
    return panel


def _annotated_frame(
    episode: Any,
    info: Mapping[str, Any],
    step: int,
    physical: Mapping[str, Any],
    source: str,
) -> np.ndarray:
    camera_position = tuple(float(value) for value in episode.sut.position)
    topdown = episode.env.render(
        mode="topdown",
        window=False,
        screen_size=(700, 700),
        scaling=5,
        camera_position=camera_position,
    )
    points = info["maneuver_reference_points_xy"]
    chase = _draw_reference_overlay(_chase_frame(episode), episode, points, view="chase")
    topdown = _draw_reference_overlay(
        np.asarray(topdown),
        episode,
        points,
        view="topdown",
        camera_position=camera_position,
    )
    phase = "lane following"
    if info.get("semantic_maneuver_active"):
        phase = "cut-in active"
    if info.get("semantic_maneuver_completed"):
        phase = "cut-in complete"
    dt = float(episode.env.config["physics_world_step_size"]) * int(episode.env.config["decision_repeat"])
    height = min(chase.shape[0], topdown.shape[0])
    chase_image = Image.fromarray(chase).resize((round(chase.shape[1] * height / chase.shape[0]), height))
    topdown_image = Image.fromarray(topdown).resize((round(topdown.shape[1] * height / topdown.shape[0]), height))
    action = info.get("raw_policy_action", (0.0, 0.0, 0.0, 0.0))
    left = _panel(
        np.asarray(chase_image),
        "Tail view | blue: SUT | red: DIVA adversary",
        f"Stage A replay | {source} | {phase} | t={step * dt:.1f} s",
    )
    right = _panel(
        np.asarray(topdown_image),
        "Top-down view | amber: effective Frenet reference",
        "gap={:.1f} m | path={:.1f} m | red request={:+.2f}".format(
            float(physical["initial_gap_m"]),
            float(physical["cutin_path_length_m"]),
            float(action[3]),
        ),
    )
    combined = Image.new("RGB", (left.width + 6 + right.width, height), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + 6, 0))
    return np.asarray(combined)


def _save_gif(frames: list[np.ndarray], output: Path) -> None:
    if not frames:
        raise RuntimeError("Stage A replay produced no frames")
    images = [Image.fromarray(frame).convert("RGB").quantize(colors=128) for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=150, loop=0, disposal=2, optimize=True)


def _render_replay(rows: list[Mapping[str, Any]], config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    candidates = [row for row in rows if row["sut_ref"] == "idm_assertive" and float(row["score"]) == 1.0]
    row = min(candidates, key=lambda value: value["design"]["design_id"])
    design = DivaCutInDesign(
        int(row["design"]["candidate_index"]),
        tuple(float(value) for value in row["design"]["logical_continuous"]),
    )
    base_task = retained_task(
        load_taskbook(config["taskbook"]), row["sut_ref"], config["study"]["task_logical_domain_id"]
    )
    task = study_task(base_task, row["logical_domain_id"], config["study"]["source_physical_bounds"])
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        design.scenario_action(),
        episode_seed=int(row["episode_seed"]),
        environment_overrides=VISUAL_OVERRIDES | {"horizon": int(config["execution"]["environment_horizon"])},
    )
    frames: list[np.ndarray] = []
    physical = row["concrete_scenario"]["initial_state"]

    def capture(current: Any, step: int, info: Mapping[str, Any]) -> None:
        if step == 0:
            current.env.engine.main_camera.track(current.sut)
            _paint_role(current.sut, (0.12, 0.43, 0.95))
            _paint_role(current.adversary, (0.92, 0.16, 0.14))
        elif step % 3 == 0:
            frames.append(_annotated_frame(current, info, step, physical, row["sut_ref"]))

    try:
        rollout = HierarchicalRunner(int(config["execution"]["runner_step_budget"])).rollout(
            episode, "cutin", step_action=DivaCutInBehavior(), step_callback=capture
        )
    finally:
        episode.env.close()
    _save_gif(frames, output)
    return {
        "source_design_id": row["design"]["design_id"],
        "source_sut": row["sut_ref"],
        "source_episode_seed": row["episode_seed"],
        "recorded_score": row["score"],
        "replay_score": float(rollout.outcome["valid_target_collision"]),
        "replay_termination_reason": rollout.outcome["termination_reason"],
        "frames": len(frames),
        "physical_parameters": physical,
    }


def run(config_path: str, observations_path: str, loso_path: str, output_dir: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows = _load_observations(Path(observations_path))
    loso = json.loads(Path(loso_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _plot_summary(rows, loso, output / "stage_a_summary.png")
    replay = _render_replay(rows, config, output / "stage_a_collision_replay.gif")
    report = {
        "schema": "diva_stage_a_visualization_v2",
        "observations": observations_path,
        "loso": loso_path,
        "summary_figure": "stage_a_summary.png",
        "collision_replay": "stage_a_collision_replay.gif",
        "replay": replay,
    }
    (output / "visualization_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/metadrive/configs/diva_cutin.yaml")
    parser.add_argument("--observations", required=True)
    parser.add_argument("--loso", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.config, args.observations, args.loso, args.output_dir)


if __name__ == "__main__":
    main()
