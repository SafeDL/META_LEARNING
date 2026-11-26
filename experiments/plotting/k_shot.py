"""PNG-only K-shot figures from complete, independently audited predictions.

Curves show equal-ADS means and pointwise ADS-bootstrap 95% intervals. Heatmaps
show paired mean NLL gains at the preregistered K=10. No simulated effect values
or outcome-based row selection are used by this entry.
"""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from experiments.history_transfer_experiment import CHECKPOINTS, OUTPUT, PREDICTION_METHODS, RUN_SCENARIOS, SEEDS
from experiments.paper_results import paper_study
from experiments.protocol import ROOT
from experiments.storage import read_json

DESTINATION = ROOT / "figures/k_shot_adaptation"
PUBLIC_SUMMARY = ROOT / "results/public_ads/reports/summary.json"
PAIRED_K = 10
PUBLIC_LABELS = (
    ("online_mc_ppo", "OnlineMC PPO*"),
    ("social_attention_ddqn", "Attention DDQN*"),
    ("decision_transformer", "Decision Transformer*"),
)
METHODS = {
    "target_only": ("Target-only GP", "#666666", "--", "o"),
    "pooled_history": ("Pooled historical GP", "#3A78A4", "-.", "^"),
    "full": ("Ours method", "#CB3434", "-", "o"),
}
SCENES = {
    "car_following_braking_onset": "Single-lane braking",
    "cut_in_onset": "Restricted cut-in",
    "cut_out_rear_gap": "Cut-out with rear traffic",
    "side_encroachment": "Side encroachment",
}
METRICS = {"nll": "NLL ↓", "brier": "Brier score ↓", "recall_100": "Collision recall@100 ↑"}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_summary(root):
    """Reject partial, substituted or changed report artifacts before plotting."""
    directory = Path(root) / "reports/main_results/prediction"
    completion = read_json(directory / "completion.json")
    protocol = read_json(directory / "protocol.json")
    required = {"prediction_summary.json", "audited_records.json", "metrics.csv"}
    if (
        completion["status"] != "complete"
        or completion["protocol"] != protocol
        or protocol["modes"] != ["prediction"]
        or completion["cases"] != 8 * len(RUN_SCENARIOS) * len(SEEDS) * len(METHODS)
        or completion["metric_rows"] != completion["cases"] * len(CHECKPOINTS)
        or not required.issubset(completion["files"])
    ):
        raise ValueError("Figures require the complete audited K-shot cohort")
    for name, digest in completion["files"].items():
        if Path(name).name != name or _sha(directory / name) != digest:
            raise ValueError("Audited figure source changed after report completion")
    summary = read_json(directory / "prediction_summary.json")
    design = summary["scopes"]["all"]["design"]
    if (
        summary["experiment"] != "strict_k_shot"
        or summary["checkpoints"] != list(CHECKPOINTS)
        or summary["scenes"] != list(RUN_SCENARIOS)
        or summary["seeds"] != list(SEEDS)
        or design["methods"] != list(PREDICTION_METHODS)
        or design["registered_ads"] != 8
    ):
        raise ValueError("Figure source does not match the registered K-shot design")
    return summary


def _curve(ax, summary, scope, metric, *, panel, scene_label=None):
    data = summary["scopes"][scope]
    index = summary["metrics"].index(metric)
    x = np.arange(len(summary["checkpoints"]))
    # Highlight the preregistered few-shot evaluation band K in {5, 10, 20, 50}.
    ax.axvspan(0.72, 4.28, color="#F4F6F8", zorder=0)
    for method, (label, color, line, marker) in METHODS.items():
        row = data["design"]["methods"].index(method)
        y, low, high = (
            np.asarray(data["unpaired"][field], dtype=float)[row, :, index]
            for field in ("mean", "ci_low", "ci_high")
        )
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=line,
            marker=marker,
            linewidth=1.45 if method == "full" else 1.15,
            markersize=4,
            markerfacecolor=color if method != "target_only" else "white",
            markeredgewidth=0.8,
        )
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
    ax.set_xticks(x, summary["checkpoints"])
    ax.set_xlim(-0.2, len(x) - 0.8)
    ax.set_ylabel(METRICS[metric])
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color="#DFE2E5", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.annotate(chr(97 + panel), (0, 1), xycoords="axes fraction",
                xytext=(0, 5), textcoords="offset points", va="bottom", fontweight="bold")
    if scene_label:
        ax.annotate(scene_label, (0, 1), xycoords="axes fraction",
                    xytext=(13, 5), textcoords="offset points", va="bottom")
    counts = np.asarray(data["unpaired"]["valid_ads"], dtype=int)[:, :, index]
    if counts.min() < data["design"]["registered_ads"]:
        low, high = counts.min(), counts.max()
        valid = str(low) if low == high else f"{low}–{high}"
        ax.set_ylabel(f"{METRICS[metric]}\n(valid ADS: {valid})")
    if not np.isfinite(np.asarray(data["unpaired"]["mean"], dtype=float)[:, :, index]).any():
        ax.text(0.5, 0.5, "Undefined: no valid tasks", transform=ax.transAxes, ha="center")
    # Include every interval; keep the probability metrics on their natural domain.
    if metric != "nll":
        low, high = ax.get_ylim()
        ax.set_ylim(max(0, low), min(1, high))
    else:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHODS),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, 0.012),
    )


def _save(fig, axes, path):
    """Check final rendered plot-area alignment and export only PNG."""
    fig.canvas.draw()
    boxes = np.asarray([ax.get_window_extent().bounds for ax in np.ravel(axes)])
    tolerance = fig.dpi * 1.5 / 72
    if np.ptp(boxes[:, 2]) > tolerance or np.ptp(boxes[:, 3]) > tolerance:
        raise ValueError("Comparable rendered plot areas are not aligned")
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    # Axis locators retain off-range tick labels that are never rendered.
    bounds = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    if (
        bounds.x0 < page.x0 - 1
        or bounds.x1 > page.x1 + 1
        or bounds.y0 < page.y0 - 1
        or bounds.y1 > page.y1 + 1
    ):
        raise ValueError("Rendered figure content falls outside the canvas")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)
    return path


def main_curves(summary, destination):
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.7), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.16, hspace=0.30)
    for panel, (ax, metric) in enumerate(zip(axes, METRICS)):
        _curve(ax, summary, "all", metric, panel=panel)
    axes[-1].set_xlabel("Target feedback checkpoint $K$")
    _legend(fig, axes[0])
    return _save(fig, axes, Path(destination) / "method_comparison_curves.png")


def scenario_curves(summary, destination, metric):
    columns = min(2, len(summary["scenes"]))
    rows = (len(summary["scenes"]) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(7.16, 3.0 * rows), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.16, wspace=0.38, hspace=0.38)
    for panel, (ax, scene) in enumerate(zip(axes.flat, summary["scenes"])):
        _curve(ax, summary, scene, metric, panel=panel, scene_label=SCENES[scene])
    fig.supxlabel("Target feedback checkpoint $K$", y=0.08, fontsize=10)
    # Use the same limits when displaying several registered scenes.
    bounds = [ax.get_ylim() for ax in axes.flat]
    for ax in axes.flat:
        ax.set_ylim(min(b[0] for b in bounds), max(b[1] for b in bounds))
    _legend(fig, axes[0, 0])
    name = "recall" if metric == "recall_100" else metric
    return _save(fig, axes, Path(destination) / f"{name}_by_scenario.png")


def _load_public_paired_gains(public_summary, scenes, k_value):
    report = read_json(public_summary)
    if report.get("completed_banks") != report.get("expected_banks") or report.get("completed_banks") != 12:
        raise ValueError("Public E01 summary is incomplete")
    lookup = {}
    for task in report["tasks"]:
        for method in ("target_only", "full"):
            entry = task["methods"][method]
            if entry["status"] != "complete":
                raise ValueError(f"Incomplete public E01 task: {task['sut_id']}/{task['scenario']}/{method}")
        to_nll = next(m["nll"] for m in task["methods"]["target_only"]["metrics"] if m["K"] == k_value)
        ours_nll = next(m["nll"] for m in task["methods"]["full"]["metrics"] if m["K"] == k_value)
        lookup[task["sut_id"], task["scenario"]] = float(to_nll) - float(ours_nll)
    rows = [[lookup[sut_id, scene] for scene in scenes] for sut_id, _ in PUBLIC_LABELS]
    return np.asarray(rows, dtype=float), [label for _, label in PUBLIC_LABELS]


def paired_gains(summary, destination, *, public_summary=None):
    k = summary["checkpoints"].index(PAIRED_K)
    metric = summary["metrics"].index("nll")
    columns = []
    for scene in summary["scenes"]:
        data = summary["scopes"][scene]
        ours = data["design"]["methods"].index("full")
        columns.append(np.asarray(data["paired"]["ads_gain"], dtype=float)[:, ours, k, metric])
    matrix = np.stack(columns, axis=1)
    ads_labels = {
        "target_idm_mobil": "IDM/MOBIL",
        "target_fvdm_lane": "FVDM",
        "target_linear_mpc": "Linear MPC",
        "target_nonlinear_trajectory": "Nonlinear",
        "target_uct": "UCT",
        "target_beam": "Beam",
        "target_ppo": "PPO",
        "target_dqn": "DQN",
    }
    names = [ads_labels[name] for name in summary["scopes"]["all"]["design"]["ads_ids"]]
    primary_rows = len(names)
    if public_summary is not None:
        public_matrix, public_names = _load_public_paired_gains(public_summary, summary["scenes"], PAIRED_K)
        matrix = np.vstack([matrix, public_matrix])
        names.extend(public_names)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()), 1e-6) if finite.size else 1.0
    height = 5.3 if public_summary is not None else 4.4
    fig, ax = plt.subplots(figsize=(7.16, height))
    fig.subplots_adjust(
        left=0.22 if public_summary is not None else 0.14,
        right=0.98,
        top=0.97 if public_summary is not None else 0.96,
        bottom=0.22 if public_summary is not None else 0.27,
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    labels = [SCENES[scene] for scene in summary["scenes"]]
    shown = ax.imshow(
        matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto"
    )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    if public_summary is not None:
        ax.axhline(primary_rows - 0.5, color="#1F2933", linewidth=1.35)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(
            col,
            row,
            "—" if not np.isfinite(value) else f"{value:+.2f}",
            ha="center",
            va="center",
            color="white" if np.isfinite(value) and abs(value) > 0.6 * limit else "#151515",
            fontsize=8,
        )
    ax.set_ylabel("Target ADS")
    color_box = (0.25, 0.085, 0.62, 0.032) if public_summary is not None else (0.22, 0.105, 0.64, 0.035)
    color_axis = fig.add_axes(color_box)
    colorbar = fig.colorbar(shown, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks(np.linspace(-limit, limit, 5))
    colorbar.set_label("ΔNLL = NLL(Target-only GP) − NLL(Ours method),  $K$ = 10", fontsize=9)
    return _save(fig, [ax], Path(destination) / "k_shot_ads_scenario_paired_gains.png")


def render(root=None, destination=DESTINATION, *, include_public=None):
    summary = load_summary(root or paper_study(OUTPUT))
    use_public = (root is None) if include_public is None else include_public
    public_summary = PUBLIC_SUMMARY if (use_public and PUBLIC_SUMMARY.is_file()) else None
    with plt.rc_context(STYLE):
        return [
            main_curves(summary, destination),
            paired_gains(summary, destination, public_summary=public_summary),
            *[scenario_curves(summary, destination, metric) for metric in METRICS],
        ]


if __name__ == "__main__":
    print(render(), flush=True)
