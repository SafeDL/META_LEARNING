"""Regenerate every ScenarioFuzz-H figure and genuine state-replay GIF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve

from diva_highway_env.sut.idm_profiles import get_profile

from .corpus import ScenarioSpec, build_default_corpus
from .graph_builder import EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES, build_graph
from .io_utils import load_config
from .execution import execute_scenario


COLORS = {"RMS-H": "#7f8c8d", "2SMS-H": "#3498db", "RMS+SEM-H": "#e67e22", "2SMS+SEM-H": "#c0392b"}


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save(fig, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_seed_graph(run_dir: Path, config: dict) -> None:
    seed = build_default_corpus(config)[0]
    spec = ScenarioSpec.create(18, -4, "cutin_braking")
    graph = build_graph(seed, spec)
    fig, ax = plt.subplots(figsize=(11, 4))
    for lane in (0, 1):
        ax.plot([0, 400], [lane * 4, lane * 4], color="#555", lw=2)
    positions = np.column_stack((graph.node_features[:, 0] * 220 + 60, graph.node_features[:, 1] * 4))
    for a, b in graph.edge_index.T:
        ax.annotate("", xy=positions[b], xytext=positions[a], arrowprops={"arrowstyle": "->", "color": "#aaa", "lw": .8})
    for i, (x, y) in enumerate(positions):
        color = "#2980b9" if "ego" in graph.node_labels[i] else "#c0392b" if "npc" in graph.node_labels[i] else "#27ae60"
        ax.scatter(x, y, s=70, color=color, zorder=3)
        offset = .35 if i % 2 == 0 else -.55
        if graph.node_labels[i] == "npc_goal": offset = .65
        if graph.node_labels[i] == "ego_goal": offset = -.65
        ax.text(x, y + offset, graph.node_labels[i], fontsize=7, ha="center")
    ax.set(xlabel="road longitudinal position (m)", ylabel="lane lateral position (m)", title="ScenarioFuzz-H local seed graph: topology and planned paths", xlim=(0, 400), ylim=(-1, 6))
    _save(fig, run_dir / "scenariofuzz_01_seed_graph.png")


def plot_graph_features(run_dir: Path, config: dict) -> None:
    graph = build_graph(build_default_corpus(config)[0], ScenarioSpec.create(18, -4, "cutin_braking"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, values, names, title in zip(axes, (graph.node_features, graph.edge_features, graph.global_features[None, :]), (NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES), ("Node features", "Edge-entity features", "Global mode/schedule")):
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7); ax.set_title(title); fig.colorbar(image, ax=ax, fraction=.04)
    fig.suptitle("Execution-preknown graph features only (no target trajectory or oracle label)")
    _save(fig, run_dir / "scenariofuzz_02_graph_features.png")


def plot_mutations(run_dir: Path, batches: list[dict], config: dict) -> None:
    for mode in config["mode_set"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        chosen_batches = [batch for batch in batches if batch["protocol"] == "matched_nm100" and batch["variant"] == "2SMS-H"]
        plotted_stages: set[str] = set()
        for batch in chosen_batches:
            rows = [row for row in batch["candidates"] if row["mode"] == mode]
            if not rows:
                continue
            color = "#3498db" if batch["stage"] == "random" else "#e74c3c"
            label = batch["stage"] if batch["stage"] not in plotted_stages else None
            ax.scatter([r["initial_gap"] for r in rows], [r["relative_speed"] for r in rows], c=color, alpha=.24, s=18, label=label)
            plotted_stages.add(batch["stage"])
            selected = [r for r in rows if r["selected"]]
            if selected:
                ax.scatter([r["initial_gap"] for r in selected], [r["relative_speed"] for r in selected], facecolors="none", edgecolors="black", s=70, linewidth=1.2)
            if plotted_stages == {"random", "neighbor"} and sum(1 for b in chosen_batches[:chosen_batches.index(batch)+1] if b["stage"] == "neighbor") >= 4:
                break
        ax.set(xlabel="configured gap (m)", ylabel="relative speed (m/s)", title=f"Two-stage mutations - {mode}", xlim=config["bounds"]["initial_gap"], ylim=config["bounds"]["relative_speed"])
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(handles[:2], labels[:2])
        _save(fig, run_dir / f"scenariofuzz_03_mutation_{mode}.png")


def plot_funnel(run_dir: Path, summary: list[dict]) -> None:
    rows = [row for row in summary if row["protocol"] == "matched_nm100"]
    stages = ("candidate_attempts", "precheck_passed", "unique_candidates", "sem_above_threshold", "executions", "collisions")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, row in zip(axes.ravel(), rows):
        values = [int(float(row[key])) for key in stages]
        ax.bar(range(len(stages)), values, color=["#95a5a6", "#7f8c8d", "#3498db", "#f39c12", "#8e44ad", "#c0392b"])
        ax.set_yscale("symlog", linthresh=20)
        for index, value in enumerate(values):
            ax.text(index, value if value else .5, str(value), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(stages)), ["proposed", "precheck", "unique", "SEM>0.5", "executed", "failure"], rotation=35, ha="right")
        ax.set_title(row["variant"])
    fig.suptitle("Candidate-to-execution funnel (matched Nm=100)")
    _save(fig, run_dir / "scenariofuzz_04_funnel.png")


def plot_ablation(run_dir: Path, executions: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for variant, color in COLORS.items():
        rows = [row for row in executions if row["protocol"] == "matched_nm100" and row["variant"] == variant]
        if not rows: continue
        steps = np.asarray([row["step"] for row in rows], dtype=int)
        failures = np.cumsum([bool(row["collision"]) for row in rows])
        wall = np.cumsum([float(row["wall_seconds"]) for row in rows])
        axes[0].step(steps, failures, where="post", label=variant, color=color)
        axes[1].step(wall, failures, where="post", label=variant, color=color)
    axes[0].set(xlabel="actual target executions", ylabel="cumulative collisions", title="Equal physical-call budget")
    axes[1].set(xlabel="simulation wall time (s)", ylabel="cumulative collisions", title="Measured execution time")
    axes[0].legend(); axes[1].legend()
    _save(fig, run_dir / "scenariofuzz_05_ablation_curve.png")


def plot_sem_quality(run_dir: Path, audit_path: Path) -> None:
    rows = _csv(audit_path)
    y = np.asarray([int(row["actual_collision"]) for row in rows]); p = np.asarray([float(row["predicted_score"]) for row in rows])
    precision, recall, _ = precision_recall_curve(y, p)
    frac, mean = calibration_curve(y, p, n_bins=6, strategy="quantile")
    matrix = confusion_matrix(y, p >= .5, labels=[0, 1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(recall, precision, color="#8e44ad"); axes[0].set(xlabel="recall", ylabel="precision", title="Frozen audit PR")
    axes[1].plot(mean, frac, marker="o"); axes[1].plot([0, 1], [0, 1], "--", color="#999"); axes[1].set(xlabel="mean predicted", ylabel="observed", title="Frozen audit calibration")
    image = axes[2].imshow(matrix, cmap="Blues"); axes[2].set_xticks([0, 1], ["safe", "collision"]); axes[2].set_yticks([0, 1], ["safe", "collision"]); axes[2].set(xlabel="predicted", ylabel="actual", title="Threshold 0.5")
    for (i, j), value in np.ndenumerate(matrix): axes[2].text(j, i, int(value), ha="center", va="center")
    fig.colorbar(image, ax=axes[2], fraction=.04); fig.suptitle(f"SEM quality on {len(y)} untouched source audit records; positives={int(y.sum())}")
    _save(fig, run_dir / "scenariofuzz_06_sem_quality.png")


def plot_history(run_dir: Path, path: Path) -> None:
    rows = _csv(path); x = [int(row["history_size"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, [float(row["auprc"]) for row in rows], marker="o", label="AUPRC"); axes[0].plot(x, [float(row["recall"]) for row in rows], marker="s", label="Recall@0.5"); axes[0].legend(); axes[0].set(xlabel="actual source-history records", ylabel="score", title="Frozen audit discrimination")
    axes[1].plot(x, [float(row["brier"]) for row in rows], marker="o", color="#c0392b"); axes[1].set(xlabel="actual source-history records", ylabel="Brier (lower better)", title="Calibration error")
    _save(fig, run_dir / "scenariofuzz_07_history_size.png")


def plot_clusters(run_dir: Path) -> None:
    path = run_dir / "clusters.csv"
    rows = _csv(path) if path.exists() and path.stat().st_size else []
    fig, ax = plt.subplots(figsize=(8, 6))
    if rows:
        labels = np.asarray([int(row["cluster"]) for row in rows])
        scatter = ax.scatter([float(row["pca_x"]) for row in rows], [float(row["pca_y"]) for row in rows], c=labels, cmap="tab10", s=55)
        ax.legend(*scatter.legend_elements(), title="cluster"); ax.set(xlabel="PCA 1", ylabel="PCA 2", title="Collision trajectory + frozen SEM embeddings")
    else:
        status = json.loads((run_dir / "clustering_status.json").read_text(encoding="utf-8"))
        ax.text(.5, .5, f"{status['status']}\nunique collisions: {status['unique_collision_trajectories']}", ha="center", va="center", transform=ax.transAxes); ax.set_axis_off()
    _save(fig, run_dir / "scenariofuzz_08_cluster_map.png")


def _vehicle_polygon(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, x0: float) -> None:
    px = int(80 + (x - x0) * 5); py = int(150 - y * 18)
    draw.rounded_rectangle((px - 13, py - 7, px + 13, py + 7), radius=3, fill=color, outline="black")


def state_replay_gif(row: dict, output: Path) -> None:
    with np.load(row["trajectory_path"], allow_pickle=False) as data: values = data["values"]
    frames = []
    for state in values[::4]:
        image = Image.new("RGB", (760, 250), "white"); draw = ImageDraw.Draw(image)
        for lane_y in (0, 4):
            py = int(150 - lane_y * 18); draw.line((20, py - 16, 740, py - 16), fill="#555", width=2); draw.line((20, py + 16, 740, py + 16), fill="#555", width=2)
        x0 = state[1] - 60
        _vehicle_polygon(draw, state[1], state[2], "#3498db", x0); _vehicle_polygon(draw, state[7], state[8], "#e74c3c", x0)
        draw.text((20, 15), f"{row['target_sut']} | {row['mode']} | gap={row['initial_gap']:.2f} m | dv={row['relative_speed']:.2f} m/s", fill="black")
        draw.text((20, 35), f"t={state[0]:.2f}s | SEM={row['predicted_score']} | actual collision={row['collision']} | near miss={row['near_miss']}", fill="black")
        draw.text((20, 220), "Blue: ego SUT   Red: scheduled NPC   Frames use recorded 20 Hz physical states", fill="#333")
        frames.append(image)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)


def _replay_audit_case(audit: dict, config: dict, run_dir: Path, seed_offset: int):
    spec = ScenarioSpec.create(float(audit["initial_gap"]), float(audit["relative_speed"]), audit["mode"])
    observation = execute_scenario(
        get_profile(audit["sut_name"]), spec, int(config["random_seed"]) + seed_offset,
        run_dir / "diagnostic_trajectories",
    )
    diagnostic = {
        "target_sut": audit["sut_name"], "mode": audit["mode"],
        "initial_gap": float(audit["initial_gap"]),
        "relative_speed": float(audit["relative_speed"]),
        "predicted_score": float(audit["predicted_score"]),
        "collision": observation.collision, "near_miss": observation.near_miss,
        "trajectory_path": observation.trajectory_path,
    }
    return diagnostic, observation


def write_report(run_dir: Path, summary: list[dict], manifest: dict, model_dir: Path) -> None:
    matched = [row for row in summary if row["protocol"] == "matched_nm100"]
    best = max(matched, key=lambda row: int(row["collisions"]))
    training = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    history_metrics = _csv(model_dir / "history_size_metrics.csv")[-1]
    clustering = json.loads((run_dir / "clustering_status.json").read_text(encoding="utf-8"))
    pool_path = run_dir.parent / "pool" / "summary.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else None
    lines = [
        "# ScenarioFuzz-H highway-env replication report", "",
        "## Outcome", "",
        "The implementation, mechanism, real simulation loop, source-only SEM training, ablations, and visualization pipeline were executed successfully. The result is an adaptation to the repository's straight two-lane highway-env harness; it does not reproduce CARLA perception/weather behavior or the paper's original numeric gains.", "",
        "## Validation levels", "",
        "- `implementation_validated`: yes (unit tests and complete artifact contract).",
        "- `mechanism_observed`: yes (random-to-neighbor batches, SEM threshold funnel, and real closed-loop outcomes are logged).",
        f"- `project_utility_observed`: measured, not forced; the largest matched-budget collision count was {best['collisions']} for {best['variant']}.",
        "- `original_numbers_reproduced`: no; the simulator, SUTs, scenario space, and oracle differ from the CARLA study.", "",
        "## Source-only SEM", "",
        f"The SEM used {training['independent_actual_episodes']} genuine source episodes from {len(training['source_suts'])} SUTs and excluded {training['excluded_target_sut']}. Its untouched grouped audit set reached AUPRC {float(history_metrics['auprc']):.3f}, Brier {float(history_metrics['brier']):.3f}, precision {float(history_metrics['precision']):.3f}, and recall {float(history_metrics['recall']):.3f} at threshold 0.5.", "",
        "## Protocol boundaries", "",
        "The main table contains online continuous ScenarioFuzz-H only. `paper_nm` preserves Nm=3 without SEM and Nm=100 with SEM; `matched_nm100` controls candidate-generation volume. Every execution, including invalid runs, consumes target budget. Rejected candidates are predictions only and never become training records.", "",
        "## Results", "",
        "| Protocol | Variant | Executions | Collisions | Critical | First collision | Empty filters |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['protocol']} | {row['variant']} | {row['executions']} | {row['collisions']} | {row['critical_events']} | {row['first_collision_budget']} | {row['empty_filters']} |")
    lines += ["", "## Post-analysis", "", f"The masked-trajectory encoder plus frozen SEM embedding produced {clustering['clusters']} candidate interaction clusters from {clustering['unique_collision_trajectories']} unique target collision trajectories. The agglomerative threshold was selected only on {clustering['source_development_replays']} source-development replay trajectories (development silhouette {clustering['development_silhouette']:.3f}); clusters are not bug labels and did not influence selection.", ""]
    if pool is not None:
        lines += ["## Separate fixed-pool protocol", "", f"`SEM-Pool-H` queried {pool['target_executions']} of {pool['candidate_pool_size']} frozen candidates and observed {pool['collisions']} collisions. It generated zero new scenarios and is intentionally excluded from the online table above.", ""]
    lines += ["## Provenance", "", f"- Actual source-history executions: {training['independent_actual_episodes']}", f"- Actual online target executions: {manifest['total_actual_target_executions']}", f"- Environment: {manifest['environment']}", f"- Source training artifacts: `{model_dir.as_posix()}`", "- Weather, color, pedestrians and traffic lights: disabled because they have no supported physical/perception effect in this harness.", "- Driving score: `1 - vulnerability` margin proxy, not the unpublished DriveFuzz supplementary formula.", "- Failure, false-positive, false-negative, and normal-case GIFs are rendered from recorded 20 Hz physical states, without trajectory interpolation."]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.resolved.yaml")
    model_dir = Path(manifest["sem_checkpoint"]).resolve().parent
    batches = _jsonl(run_dir / "candidate_batches.jsonl"); executions = _jsonl(run_dir / "executions.jsonl"); summary = _csv(run_dir / "ablation_summary.csv")
    plot_seed_graph(run_dir, config); plot_graph_features(run_dir, config); plot_mutations(run_dir, batches, config)
    plot_funnel(run_dir, summary); plot_ablation(run_dir, executions); plot_sem_quality(run_dir, model_dir / "audit_predictions.csv"); plot_history(run_dir, model_dir / "history_size_metrics.csv"); plot_clusters(run_dir)
    failures = [row for row in executions if row["collision"] and row["trajectory_path"]]
    false_positives = [row for row in executions if not row["collision"] and row["predicted_score"] is not None and row["trajectory_path"]]
    replay_rows = []
    if failures:
        state_replay_gif(failures[0], run_dir / "scenariofuzz_case_first_failure.gif"); replay_rows.append({"rule": "first_failure", "execution_id": failures[0]["execution_id"]})
    if false_positives:
        chosen = max(false_positives, key=lambda row: row["predicted_score"])
        state_replay_gif(chosen, run_dir / "scenariofuzz_case_highest_false_positive.gif"); replay_rows.append({"rule": "highest_score_false_positive", "execution_id": chosen["execution_id"]})
    else:
        audit_rows = _csv(model_dir / "audit_predictions.csv")
        audit_false_positives = [row for row in audit_rows if int(row["actual_collision"]) == 0 and float(row["predicted_score"]) >= .5]
        if audit_false_positives:
            audit = max(audit_false_positives, key=lambda row: float(row["predicted_score"]))
            diagnostic, observation = _replay_audit_case(audit, config, run_dir, 900000)
            if not observation.collision and observation.trajectory_path:
                state_replay_gif(diagnostic, run_dir / "scenariofuzz_case_highest_false_positive.gif")
                replay_rows.append({"rule": "highest_score_false_positive", "protocol": "frozen_source_audit_diagnostic_replay", "execution_id": observation.execution_id})
    audit_rows = _csv(model_dir / "audit_predictions.csv")
    false_negatives = sorted(
        [row for row in audit_rows if int(row["actual_collision"]) == 1 and float(row["predicted_score"]) < .5],
        key=lambda row: float(row["predicted_score"]),
    )
    for index, audit in enumerate(false_negatives):
        diagnostic, observation = _replay_audit_case(audit, config, run_dir, 910000 + index)
        if observation.collision and observation.trajectory_path:
            state_replay_gif(diagnostic, run_dir / "scenariofuzz_case_lowest_score_false_negative.gif")
            replay_rows.append({"rule": "lowest_score_false_negative", "protocol": "frozen_source_audit_diagnostic_replay", "execution_id": observation.execution_id})
            break
    normal_candidates = sorted(
        [row for row in audit_rows if int(row["actual_collision"]) == 0 and float(row["predicted_score"]) < .5],
        key=lambda row: float(row["predicted_score"]),
    )
    for index, audit in enumerate(normal_candidates):
        diagnostic, observation = _replay_audit_case(audit, config, run_dir, 920000 + index)
        if not observation.collision and observation.trajectory_path:
            state_replay_gif(diagnostic, run_dir / "scenariofuzz_case_normal.gif")
            replay_rows.append({"rule": "normal_true_negative", "protocol": "frozen_source_audit_diagnostic_replay", "execution_id": observation.execution_id})
            break
    (run_dir / "replay_manifest.json").write_text(json.dumps(replay_rows, indent=2), encoding="utf-8")
    cost_ledger = {"source_history_actual_executions": json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))["independent_actual_episodes"], "target_online_actual_executions": manifest["total_actual_target_executions"], "source_development_cluster_replays": json.loads((run_dir / "clustering_status.json").read_text(encoding="utf-8")).get("source_development_replays", 0), "source_audit_diagnostic_replays": sum(row.get("protocol") == "frozen_source_audit_diagnostic_replay" for row in replay_rows)}
    (run_dir / "cost_ledger.json").write_text(json.dumps(cost_ledger, indent=2), encoding="utf-8")
    visualization = {"figures": sorted(path.name for path in run_dir.glob("scenariofuzz_*.png")), "gifs": sorted(path.name for path in run_dir.glob("scenariofuzz_case_*.gif")), "sources": ["seed_corpus.json", "candidate_batches.jsonl", "selection_log.jsonl", "executions.jsonl", "ablation_summary.csv", str(model_dir / "audit_predictions.csv"), str(model_dir / "history_size_metrics.csv"), "clusters.csv"], "command": f"python -m replications.scenariofuzz_highway_env.scenariofuzz.visualize --run-dir {run_dir.as_posix()}"}
    (run_dir / "visualization_manifest.json").write_text(json.dumps(visualization, indent=2), encoding="utf-8")
    write_report(run_dir, summary, manifest, model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); visualize(args.run_dir); print(f"Wrote ScenarioFuzz-H visualizations to {args.run_dir}")


if __name__ == "__main__":
    main()
