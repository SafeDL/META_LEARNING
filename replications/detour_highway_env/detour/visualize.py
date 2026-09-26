"""Plots and genuine highway-env GIF replays for DETOUR-Scenario-H runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from highway_sim_env.envs.cutin_env import CutInEnv, CutInScenario
from replications.detour_highway_env.detour.features import RoadCurvatureFeatures
from replications.detour_highway_env.detour.contracts import ScenarioSpec
from sut_algorithms.highway_env.idm_profiles import get_profile


def plot_road_features(path: Path) -> None:
    """Plot a synthetic road and its curvature reconstruction for a unit check."""
    x = np.linspace(0.0, 60.0, 61)
    original = np.column_stack((x, 4.0 * np.sin(x / 12.0)))
    features = RoadCurvatureFeatures.from_points(original)
    compressed = features.compressed(12)
    rebuilt = compressed.reconstruct(original[0])
    figure, (road_axis, curvature_axis) = plt.subplots(1, 2, figsize=(11, 4))
    road_axis.plot(original[:, 0], original[:, 1], label="original points", linewidth=2)
    road_axis.plot(rebuilt[:, 0],
                   rebuilt[:, 1],
                   "--",
                   label="12-segment compressed reconstruction")
    road_axis.set(xlabel="x (m)", ylabel="y (m)", title="Original-road feature unit check")
    road_axis.legend()
    curvature_axis.plot(features.curvatures, color="#9b3a2e")
    curvature_axis.set(xlabel="road segment",
                       ylabel="curvature",
                       title="Extracted curvature before greedy compression")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_tree(path: Path, tree, trace) -> None:
    """Render a compact actual retrieval path through the Ward hierarchy."""
    leaves = sorted((node for node in tree.nodes.values() if node.left is None),
                    key=lambda node: node.node_id)
    y_values = {node.node_id: float(index) for index, node in enumerate(leaves)}

    def assign(node):
        if node.left is None: return y_values[node.node_id]
        y_values[node.node_id] = (assign(node.left) + assign(node.right)) / 2.0
        return y_values[node.node_id]

    assign(tree.root)
    depths = {}

    def depth(node, value=0):
        depths[node.node_id] = value
        if node.left is not None:
            depth(node.left, value + 1)
            depth(node.right, value + 1)

    depth(tree.root)
    figure, axis = plt.subplots(figsize=(11, 8))
    for node in tree.nodes.values():
        if node.left is None: continue
        x, left, right = depths[node.node_id], node.left, node.right
        axis.plot([x, x], [y_values[left.node_id], y_values[right.node_id]],
                  color="0.75",
                  linewidth=.5)
        axis.plot([x, depths[left.node_id]], [y_values[left.node_id], y_values[left.node_id]],
                  color="0.75",
                  linewidth=.5)
        axis.plot([x, depths[right.node_id]], [y_values[right.node_id], y_values[right.node_id]],
                  color="0.75",
                  linewidth=.5)
    path_ids = set(trace.path_node_ids) if trace else set()
    for node_id in path_ids:
        axis.scatter(depths[node_id], y_values[node_id], c="#d62728", s=22, zorder=3)
    failure_y = [y_values[index] for index in tree.failure_indices]
    candidate_y = [y_values[tree.history_count + index] for index in range(tree.candidate_count)]
    axis.scatter([depths[index] for index in tree.failure_indices],
                 failure_y,
                 marker="x",
                 c="#b2182b",
                 s=24,
                 label="failing history")
    axis.scatter([depths[tree.history_count + index] for index in range(tree.candidate_count)],
                 candidate_y,
                 c="#2166ac",
                 s=8,
                 label="candidates")
    axis.set(title="Ward hierarchy and one DETOUR retrieval path",
             xlabel="tree depth",
             ylabel="leaf ordering")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_selection(path: Path, specs: tuple[ScenarioSpec, ...], selected: list[int],
                   known_failure_specs: tuple[ScenarioSpec, ...]) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter([item.initial_gap for item in specs], [item.relative_speed for item in specs],
                 s=18,
                 color="0.7",
                 label="candidate pool")
    failures = list(known_failure_specs)
    axis.scatter([item.initial_gap for item in failures],
                 [item.relative_speed for item in failures],
                 marker="x",
                 s=35,
                 color="#b2182b",
                 label="known source failures")
    chosen = [specs[index] for index in selected]
    scatter = axis.scatter([item.initial_gap for item in chosen],
                           [item.relative_speed for item in chosen],
                           c=np.arange(1,
                                       len(chosen) + 1),
                           cmap="viridis",
                           s=42,
                           label="DETOUR order")
    figure.colorbar(scatter, ax=axis, label="selection order")
    axis.set(xlabel="configured gap (m)",
             ylabel="relative speed (m/s)",
             title="Input-only candidate selection")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_discovery_curve(path: Path, selected_path: Path) -> None:
    rows = _read_csv(selected_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, color in (("DETOUR-static", "#d62728"), ("Random", "#666666"),
                          ("Nearest-Failure-global", "#2166ac")):
        subset = [row for row in rows if row["method"] == method]
        grouped: dict[int, list[int]] = {}
        for row in subset:
            grouped.setdefault(int(row["step"]), []).append(int(row["cumulative_failures"]))
        steps = sorted(grouped)
        means = np.array([np.mean(grouped[step]) for step in steps])
        std = np.array([np.std(grouped[step]) for step in steps])
        axis.plot(steps, means, label=method, color=color)
        axis.fill_between(steps, means - std, means + std, color=color, alpha=.15)
    axis.set(xlabel="target queries",
             ylabel="cumulative collision discoveries",
             title="Offline discovery curve (target truth shown only after selection)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_sensitivity(path: Path, sensitivity_path: Path) -> None:
    rows = _read_csv(sensitivity_path)
    m_values = sorted({int(row["m_neighbors"])
                       for row in rows})
    w_values = sorted({int(row["w_streak"])
                       for row in rows})
    ratio_pairs = sorted({(row["min_ratio"], row["max_ratio"])
                          for row in rows},
                         key=lambda pair: tuple(map(float, pair)))
    figure, axes = plt.subplots(3,
                                len(ratio_pairs),
                                figsize=(4 * len(ratio_pairs), 10),
                                constrained_layout=True,
                                squeeze=False)
    for row_index, (field, title) in enumerate(
        (("selected_count", "selected count"), ("failure_ratio", "collision failure ratio"),
         ("recall", "collision recall"))):
        for column, (minimum, maximum) in enumerate(ratio_pairs):
            axis = axes[row_index, column]
            values = np.array([[
                np.nanmean([
                    float(row[field]) for row in rows
                    if int(row["m_neighbors"]) == m and int(row["w_streak"]) == w
                    and row["min_ratio"] == minimum and row["max_ratio"] == maximum
                ]) for w in w_values
            ] for m in m_values])
            image = axis.imshow(values, cmap="viridis")
            figure.colorbar(image, ax=axis)
            axis.set(xticks=range(len(w_values)),
                     xticklabels=w_values,
                     yticks=range(len(m_values)),
                     yticklabels=m_values,
                     xlabel="w safe-neighbor streak",
                     ylabel="m nearest histories",
                     title=f"{title}; min/max={minimum}/{maximum}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_branch_statistics(path: Path, trace_path: Path) -> None:
    rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    expected, observed = [], []
    for row in rows:
        path_ids = row["path_node_ids"]
        for index, probabilities in enumerate(row["branch_probabilities"]):
            chosen = str(path_ids[index + 1])
            for child_id, probability in probabilities.items():
                expected.append(float(probability))
                observed.append(float(child_id == chosen))
    figure, axis = plt.subplots(figsize=(6, 4))
    if expected:
        bins = np.linspace(0.0, 1.0, 11)
        index = np.digitize(expected, bins, right=True)
        x, y = [], []
        for bin_index in sorted(set(index)):
            mask = np.asarray(index) == bin_index
            x.append(float(np.mean(np.asarray(expected)[mask])))
            y.append(float(np.mean(np.asarray(observed)[mask])))
        axis.scatter(x, y, color="#4c78a8", label="repeated branch choices")
        axis.plot([0, 1], [0, 1], "--", color="0.35", label="expected = observed")
    axis.set(xlabel="expected failure-ratio branch probability",
             ylabel="empirical selection frequency",
             title="Recorded DETOUR branch sampling")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_replay_gif(path: Path,
                     target_sut: str,
                     scenario: ScenarioSpec,
                     seed: int,
                     node_id: str = "offline replay") -> dict[str, object]:
    """Capture actual CutInEnv rgb_array frames, annotate them, and save a GIF."""
    environment = CutInEnv(get_profile(target_sut),
                           CutInScenario(scenario.initial_gap, scenario.relative_speed,
                                         scenario.mode),
                           render_mode="rgb_array")
    frames = []
    try:
        environment.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, _info = environment.step(1)
            frame = Image.fromarray(environment.render()).convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((4, 4, 790, 51), fill=(255, 255, 255))
            draw.text(
                (8, 7),
                f"version/SUT={target_sut} | node={node_id} | t={environment.time:.1f}s | {scenario.scenario_id}",
                fill=(0, 0, 0))
            draw.text(
                (8, 27),
                f"mode={scenario.mode} | configured gap={scenario.initial_gap:.2f} m | relative speed={scenario.relative_speed:.2f} m/s",
                fill=(0, 0, 0))
            frames.append(frame)
        result = environment.episode_result()
    finally:
        environment.close()
    if not frames: raise RuntimeError("CutInEnv produced no replay frames")
    draw = ImageDraw.Draw(frames[-1])
    draw.rectangle((4, 52, 450, 72), fill=(255, 245, 245))
    draw.text(
        (8, 56),
        f"event: collision={result.collision}, near_miss={result.near_miss}, completed={result.completed}",
        fill=(120, 0, 0))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    return {
        "collision": result.collision,
        "near_miss": result.near_miss,
        "completed": result.completed,
        "frames": len(frames)
    }
