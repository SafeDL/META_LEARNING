"""Rebuild FST figures and real highway-env replay GIFs from run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

from diva_highway_env.data.response_bank import ResponseBank
from diva_highway_env.envs.cutin_env import CutInEnv, CutInScenario
from diva_highway_env.sut.idm_profiles import get_profile


COLORS = {
    "CMC": "#9c9c9c",
    "Uniform": "#72b7b2",
    "IS": "#f2cf5b",
    "Handcrafted-Similarity": "#f28e2b",
    "FST-RandomSet": "#b279a2",
    "FST-Similarity-H": "#4e79a7",
    "NoSimilarity-Optimized": "#e15759",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _scatter_axes(axis: plt.Axes, candidates: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    speed = np.asarray([float(row["relative_speed"]) for row in candidates])
    gap = np.asarray([float(row["configured_gap"]) for row in candidates])
    axis.set_xlabel("Relative speed (m/s), lead - ego")
    axis.set_ylabel("Configured gap (m)")
    axis.grid(alpha=0.15)
    return speed, gap


def _plot_reference(run_dir: Path, candidates: list[dict[str, str]]) -> Path:
    speed = np.asarray([float(row["relative_speed"]) for row in candidates])
    gap = np.asarray([float(row["configured_gap"]) for row in candidates])
    probability_rows = _read_csv(run_dir / "reference_distribution.csv")
    p = np.asarray([float(row["probability"]) for row in probability_rows])
    collision_columns = [key for key in candidates[0] if key.startswith("collision_")]
    collision_count = np.asarray([
        sum(int(row[key]) for key in collision_columns) for row in candidates
    ])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis in axes:
        _scatter_axes(axis, candidates)
    scatter = axes[0].scatter(speed, gap, c=p, cmap="viridis", s=46, edgecolor="black", linewidth=0.25)
    figure.colorbar(scatter, ax=axes[0], label="Reference probability mass p(x)")
    axes[0].set_title("Uniform benchmark reference distribution")
    scatter = axes[1].scatter(
        speed, gap, c=collision_count, cmap="magma", vmin=0, vmax=len(collision_columns),
        s=46, edgecolor="black", linewidth=0.25,
    )
    figure.colorbar(scatter, ax=axes[1], label="Number of source SUT collisions")
    axes[1].set_title("Measured source responses (no interpolation)")
    mode = candidates[0]["mode"] if len(set(row["mode"] for row in candidates)) == 1 else "joint"
    path = run_dir / f"fst_01_reference_distribution_{mode}.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _plot_similarity(run_dir: Path, candidates: list[dict[str, str]], arrays: dict[str, np.ndarray], weights: list[dict[str, str]]) -> Path:
    available = sorted(int(key.split("n")[-1]) for key in arrays if key.startswith("S_n"))
    budget = 10 if 10 in available else available[len(available) // 2]
    selected = arrays[f"selected_n{budget}"].astype(int)
    rows = [row for row in weights if int(row["budget"]) == budget]
    focal_position = int(np.argmax([float(row["weight"]) for row in rows]))
    focal_index = int(selected[focal_position])
    similarity = arrays[f"S_n{budget}"][focal_position]
    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    speed, gap = _scatter_axes(axis, candidates)
    scatter = axis.scatter(speed, gap, c=similarity, cmap="turbo", vmin=0, vmax=1, s=55)
    axis.scatter(speed[focal_index], gap[focal_index], marker="*", s=230, facecolor="white", edgecolor="black")
    axis.set_title(f"Learned similarity from focal selected scenario (n={budget})")
    figure.colorbar(scatter, ax=axis, label=r"$S_{i\ell}$ (fixed 0-1 scale)")
    scenario_id = candidates[focal_index]["scenario_id"]
    path = run_dir / f"fst_02_similarity_point_{scenario_id}.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _plot_partitions_and_weights(
    run_dir: Path,
    candidates: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    weight_rows: list[dict[str, str]],
) -> list[Path]:
    paths: list[Path] = []
    speed = np.asarray([float(row["relative_speed"]) for row in candidates])
    gap = np.asarray([float(row["configured_gap"]) for row in candidates])
    collision_columns = [key for key in candidates[0] if key.startswith("collision_")]
    for key in sorted((key for key in arrays if key.startswith("S_n")), key=lambda value: int(value[3:])):
        budget = int(key[3:])
        attention = arrays[key]
        selected = arrays[f"selected_n{budget}"].astype(int)
        partition = attention.argmax(axis=0)
        figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        axis.scatter(speed, gap, c=partition, cmap="tab20", s=55, alpha=0.9)
        boundary = np.asarray([
            0 < sum(int(row[column]) for column in collision_columns) < len(collision_columns)
            for row in candidates
        ])
        axis.scatter(speed[boundary], gap[boundary], facecolor="none", edgecolor="black", s=85, linewidth=0.8)
        axis.scatter(speed[selected], gap[selected], marker="x", c="black", s=90, linewidth=1.8)
        axis.set_title(f"Learned soft partition by argmax similarity (n={budget})")
        axis.set_xlabel("Relative speed (m/s), lead - ego")
        axis.set_ylabel("Configured gap (m)")
        axis.grid(alpha=0.15)
        path = run_dir / f"fst_03_soft_partition_n{budget}.png"
        figure.savefig(path, dpi=190)
        plt.close(figure)
        paths.append(path)

        rows = sorted(
            (row for row in weight_rows if int(row["budget"]) == budget),
            key=lambda row: int(row["position"]),
        )
        values = np.asarray([float(row["weight"]) for row in rows])
        counts = np.asarray([int(row["source_collision_count"]) for row in rows])
        figure, axis = plt.subplots(figsize=(max(6.4, budget * 0.42), 4.2), constrained_layout=True)
        bars = axis.bar(np.arange(budget), values, color=plt.cm.magma(counts / max(1, len(collision_columns))))
        cumulative = np.cumsum(values)
        second = axis.twinx()
        second.plot(np.arange(budget), cumulative, color="#2f4b7c", marker="o", label="cumulative mass")
        axis.set(xlabel="Selected-set position", ylabel="Representative mass weight", title=f"Frozen FST weights (n={budget})")
        second.set_ylabel("Cumulative probability mass")
        second.set_ylim(0, 1.05)
        for bar, count in zip(bars, counts):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=8)
        path = run_dir / f"fst_04_weights_n{budget}.png"
        figure.savefig(path, dpi=190)
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_errors(run_dir: Path, rows: list[dict[str, str]]) -> Path:
    methods = list(COLORS)
    budgets = sorted(set(int(row["budget"]) for row in rows))
    figure, axes = plt.subplots(1, len(budgets), figsize=(6.2 * len(budgets), 5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, budget in zip(axes, budgets):
        values = [
            [float(row["absolute_error"]) for row in rows if row["method"] == method and int(row["budget"]) == budget]
            for method in methods
        ]
        plot = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, method in zip(plot["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.75)
        zero_counts = [sum(abs(value) < 1e-12 for value in group) for group in values]
        axis.set_title(f"n={budget}; zero errors: {zero_counts}")
        axis.set_xticks(np.arange(1, len(methods) + 1), methods, rotation=58, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Absolute collision-rate estimation error")
    figure.suptitle("Held-out target errors over offline algorithm replays")
    path = run_dir / "fst_05_estimation_error.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _plot_ideal(run_dir: Path, rows: list[dict[str, str]]) -> Path:
    figure, axis = plt.subplots(figsize=(6.2, 5), constrained_layout=True)
    for budget in sorted(set(int(row["budget"]) for row in rows)):
        subset = [row for row in rows if int(row["budget"]) == budget]
        axis.scatter(
            [float(row["max_source_error"]) for row in subset],
            [float(row["ideal_convex_target_error"]) for row in subset],
            s=28, alpha=0.65, label=f"n={budget}",
        )
    limit = max(max(float(row["max_source_error"]), float(row["ideal_convex_target_error"])) for row in rows)
    axis.plot([0, limit], [0, limit], "--", color="black", label="convex-hull upper bound")
    axis.set(xlabel="Maximum source-model error", ylabel="Artificial convex-target error", title="Algebraic ideal-target bound (not a real AV)")
    axis.legend()
    axis.grid(alpha=0.2)
    path = run_dir / "fst_06_ideal_bound.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _plot_search(run_dir: Path, rows: list[dict[str, str]]) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    methods = ["FST-Similarity-H", "Handcrafted-Similarity", "NoSimilarity-Optimized"]
    for budget in sorted(set(int(row["budget"]) for row in rows)):
        for method in methods:
            subset = [row for row in rows if row["method"] == method and int(row["budget"]) == budget]
            axis.step(
                [int(row["candidate_evaluations"]) for row in subset],
                [float(row["loss"]) for row in subset],
                where="post", label=f"{method}, n={budget}", alpha=0.8,
            )
    axis.set(xlabel="Cumulative source-only candidate evaluations", ylabel="Source minimax loss", title="Offline set search (no target loss used)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    path = run_dir / "fst_07_search_loss.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _plot_budget_generalization(run_dir: Path, rows: list[dict[str, str]], train_n: int) -> Path:
    budgets = sorted(set(int(row["budget"]) for row in rows))
    matrix = np.asarray([[
        np.mean([
            float(row["absolute_error"]) for row in rows
            if row["method"] == "FST-Similarity-H" and int(row["budget"]) == budget
        ]) for budget in budgets
    ]])
    figure, axis = plt.subplots(figsize=(6.2, 2.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    for column, value in enumerate(matrix[0]):
        axis.text(column, 0, f"{value:.4f}", ha="center", va="center", color="black")
    axis.set_xticks(range(len(budgets)), budgets)
    axis.set_yticks([0], [f"train_n={train_n}"])
    axis.set_xlabel("Frozen set size used for evaluation")
    axis.set_title("Budget generalization of one frozen similarity network")
    figure.colorbar(image, ax=axis, label="Mean absolute held-out error")
    path = run_dir / "fst_08_budget_generalization.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _render_gif(
    path: Path,
    sut_name: str,
    candidate: dict[str, str],
    seed: int,
    role: str,
) -> dict[str, object]:
    scenario = CutInScenario(
        float(candidate["configured_gap"]),
        float(candidate["relative_speed"]),
        candidate["mode"],
    )
    environment = CutInEnv(get_profile(sut_name), scenario, render_mode="rgb_array")
    frames: list[Image.Image] = []
    try:
        environment.reset(seed=seed + int(candidate["scenario_index"]))
        terminated = truncated = False
        while not (terminated or truncated):
            _observation, _reward, terminated, truncated, _info = environment.step(1)
            frame = Image.fromarray(environment.render()).convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((3, 3, min(790, frame.width - 3), 52), fill=(255, 255, 255))
            draw.text((8, 7), f"FST {role} | target={sut_name} | t={environment.time:.1f}s", fill=(0, 0, 0))
            draw.text(
                (8, 28),
                f"{candidate['scenario_id']} | gap={float(candidate['configured_gap']):.2f} m | rel.speed={float(candidate['relative_speed']):.2f} m/s",
                fill=(0, 0, 0),
            )
            frames.append(frame)
        result = environment.episode_result()
    finally:
        environment.close()
    if not frames:
        raise RuntimeError("highway-env produced no RGB frames")
    draw = ImageDraw.Draw(frames[-1])
    draw.rectangle((3, 53, min(530, frames[-1].width - 3), 75), fill=(255, 245, 245))
    draw.text((8, 57), f"collision={result.collision} | near_miss={result.near_miss} | completed={result.completed}", fill=(120, 0, 0))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    return {
        "file": path.name,
        "role": role,
        "target_sut": sut_name,
        "scenario_id": candidate["scenario_id"],
        "configured_gap": float(candidate["configured_gap"]),
        "relative_speed": float(candidate["relative_speed"]),
        "mode": candidate["mode"],
        "collision": bool(result.collision),
        "near_miss": bool(result.near_miss),
        "completed": bool(result.completed),
        "frames": len(frames),
        "source": "actual CutInEnv(render_mode='rgb_array') frames",
    }


def visualize(run_dir: Path) -> dict[str, object]:
    candidates = _read_csv(run_dir / "candidate_table.csv")
    weights = _read_csv(run_dir / "weights.csv")
    estimates = _read_csv(run_dir / "target_estimates.csv")
    ideal = _read_csv(run_dir / "ideal_bound.csv")
    search = _read_csv(run_dir / "search_trace.csv")
    arrays_file = np.load(run_dir / "S.npz", allow_pickle=False)
    arrays = {key: arrays_file[key] for key in arrays_file.files}
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    outputs = [str(_plot_reference(run_dir, candidates).name)]
    outputs.append(str(_plot_similarity(run_dir, candidates, arrays, weights).name))
    outputs.extend(path.name for path in _plot_partitions_and_weights(run_dir, candidates, arrays, weights))
    outputs.append(_plot_errors(run_dir, estimates).name)
    outputs.append(_plot_ideal(run_dir, ideal).name)
    outputs.append(_plot_search(run_dir, search).name)
    outputs.append(_plot_budget_generalization(run_dir, estimates, int(config["train_n"])).name)

    budget = max(int(key[3:]) for key in arrays if key.startswith("S_n"))
    selected = arrays[f"selected_n{budget}"].astype(int)
    rows = sorted(
        (row for row in weights if int(row["budget"]) == budget),
        key=lambda row: int(row["position"]),
    )
    weight_values = np.asarray([float(row["weight"]) for row in rows])
    source_counts = np.asarray([int(row["source_collision_count"]) for row in rows])
    safe_positions = np.flatnonzero(source_counts == 0)
    safe_position = int(safe_positions[np.argmax(weight_values[safe_positions])]) if len(safe_positions) else int(np.argmax(weight_values))
    boundary_positions = np.flatnonzero(source_counts > 0)
    boundary_position = int(boundary_positions[np.argmin(weight_values[boundary_positions])]) if len(boundary_positions) else int(np.argmin(weight_values))
    target = str(config["target_suts"][0])
    safe_candidate = candidates[int(selected[safe_position])]
    boundary_candidate = candidates[int(selected[boundary_position])]
    replays = [
        _render_gif(run_dir / "fst_case_safe.gif", target, safe_candidate, int(config["seed"]), "high-weight safe representative"),
        _render_gif(run_dir / "fst_case_boundary.gif", target, boundary_candidate, int(config["seed"]), "low-weight source-boundary representative"),
    ]
    outputs.extend(["fst_case_safe.gif", "fst_case_boundary.gif"])
    result = {
        "status": "visualization_complete",
        "input_artifacts": [
            "candidate_table.csv", "reference_distribution.csv", "S.npz", "weights.csv",
            "target_estimates.csv", "ideal_bound.csv", "search_trace.csv",
        ],
        "plots": outputs,
        "replays": replays,
        "bank_sha256": manifest["bank_sha256"],
        "rendering_note": (
            f"scatter values are measured at all {len(candidates)} anchors; "
            "no interpolated field is presented as truth"
        ),
    }
    (run_dir / "visualization_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(visualize(args.run_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
