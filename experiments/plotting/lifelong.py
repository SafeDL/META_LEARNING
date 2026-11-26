"""Export the unified 12-ADS evidence figures as audited 600 dpi PNGs."""

import string

import numpy as np

from experiments.k_shot import finite_mean
from experiments.lifelong.report import REPORT, verify_report_links
from experiments.plotting.style import STYLE, audit_figure, plt
from experiments.protocol import ROOT
from experiments.storage import read_json

FIGURES = ROOT / "figures/lifelong_transfer"
LABELS = {
    "target_idm_mobil": "IDM/MOBIL", "target_fvdm_lane": "FVDM",
    "target_linear_mpc": "Linear MPC", "target_nonlinear_trajectory": "Nonlinear trajectory",
    "target_uct": "UCT", "target_beam": "Beam", "target_ppo": "PPO",
    "target_dqn": "Continual DQN", "online_mc_ppo": "OnlineMC PPO",
    "social_attention_ddqn": "Social Attention DDQN", "decision_transformer": "Decision Transformer",
    "continuous_sac": "Continuous SAC", "interval_robust_planner": "Interval Robust Planner",
    "full": "Ours v1", "lifelong_full": "Ours method", "target_only": "Target-only GP",
    "pooled_history": "Pooled GP", "static_prior": "Static prior", "random": "Random",
    "scenario_fuzz": "ScenarioFuzz", "paired_system_active_evaluation": "Paired system",
    "local_reliability": "+LocalRel", "risk_ucb": "+Risk-UCB", "frozen": "Frozen weights",
    "fixed": "Fixed two sources", "naive_append": "Naive append",
    "debiased_append": "Debiased append", "full_lifelong": "Ours method",
}
COLORS = {
    "lifelong_full": "#0B5368", "full_lifelong": "#0B5368", "full": "#779198",
    "target_only": "#AD6435", "pooled_history": "#806C9E", "random": "#777777",
    "scenario_fuzz": "#5D8457", "paired_system_active_evaluation": "#AA5570",
    "static_prior": "#6B7999", "frozen": "#777777", "local_reliability": "#43868E",
    "risk_ucb": "#BE8B58", "fixed": "#777777",
    "naive_append": "#BE8B58", "debiased_append": "#43868E",
}
STYLES = {
    "lifelong_full": "-", "full_lifelong": "-", "full": "--", "target_only": "-.",
    "pooled_history": ":", "static_prior": ":", "random": "--", "frozen": "--",
    "scenario_fuzz": (0, (2, 2)), "paired_system_active_evaluation": (0, (4, 2, 1, 2)),
    "local_reliability": "--", "risk_ucb": "-.", "fixed": ":",
    "naive_append": "--", "debiased_append": "-.",
}
SMALL = {**STYLE, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "legend.fontsize": 7}


def _panels(rows, columns, *, height, bottom=0.13, **kwargs):
    fig, axes = plt.subplots(rows, columns, figsize=(7.1, height), squeeze=False, **kwargs)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=bottom, top=0.91, wspace=0.39, hspace=0.48)
    if len(axes.flat) > 1:
        for i, ax in enumerate(axes.flat):
            ax.text(-0.14, 1.04, string.ascii_lowercase[i], transform=ax.transAxes,
                    ha="right", va="bottom", weight="bold")
    return fig, axes


def _line(ax, x, y, method, **kwargs):
    main = method in ("lifelong_full", "full_lifelong")
    return ax.plot(
        x, y, color=COLORS[method], linestyle=STYLES[method], label=LABELS[method],
        linewidth=1.5 if main else 0.9, zorder=4 if main else 2, **kwargs,
    )[0]


def _legend(fig, ax, columns):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.008),
               ncol=columns, frameon=False, handlelength=2.6)


def _save(fig, axes, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit_figure(fig, list(axes.flat), name, report_root=REPORT / "figures")
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    plt.close(fig)


def discovery(report):
    budget = report["budget"]
    curves = np.asarray(budget["curves"]).mean(1)
    methods = (*budget["methods"][:7], "lifelong_full")
    fig, axes = _panels(3, 4, height=5.65, bottom=0.14, sharex=True, sharey=True)
    for t, (target, ax) in enumerate(zip(report["targets"], axes.flat, strict=True)):
        ax.set_title(LABELS[target], fontsize=7, pad=3)
        for method in methods:
            _line(ax, np.arange(1, 201), curves[t, budget["methods"].index(method)], method)
        ax.set_xlim(0, 200)
        ax.set_xticks((0, 100, 200))
        ax.set_ylim(0, 200)
        ax.set_yticks((0, 100, 200))
    for ax in axes[-1]:
        ax.set_xlabel("Paid queries")
    for ax in axes[:, 0]:
        ax.set_ylabel("Collisions found")
    _legend(fig, axes[0, 0], 4)
    _save(fig, axes, "discovery_by_ads")


def composition(report):
    budget = report["budget"]
    values = np.asarray(budget["region_counts"]).mean(1)
    methods = ("target_only", "full", "lifelong_full")
    region_colors = ("#40607A", "#8FAFC2", "#C97A4E")
    labels = ("All eight in-family ADSs", "Some in-family ADSs", "No in-family ADS")
    fig, axes = _panels(3, 4, height=5.25, bottom=0.16, sharex=True, sharey=True)
    for t, (target, ax) in enumerate(zip(report["targets"], axes.flat, strict=True)):
        ax.set_title(LABELS[target], fontsize=7, pad=3)
        levels = values[t, [budget["methods"].index(method) for method in methods]]
        bottom = np.zeros(len(methods))
        for r, (label, color) in enumerate(zip(labels, region_colors, strict=True)):
            heights = levels[:, r]
            ax.bar(np.arange(len(methods)), heights, bottom=bottom, width=0.68, color=color, label=label)
            bottom += heights
        ax.set_ylim(0, 200)
        ax.set_yticks((0, 100, 200))
        ax.set_xticks(range(len(methods)), ("Target-only\nGP", "v1", "Ours\nmethod"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Collisions found")
    _legend(fig, axes[0, 0], 3)
    _save(fig, axes, "discovery_region_composition")


def regions(report):
    budget = report["budget"]
    n = len(report["targets"]) - 4
    values = np.asarray(budget["region_counts"])[n:].mean(1)
    methods = ("target_only", "full", "local_reliability", "risk_ucb", "lifelong_full")
    fig, axes = _panels(1, 2, height=2.95, bottom=0.29, sharey=True)
    fig.subplots_adjust(top=0.82)
    last_label = "Continuous\nSAC" if report["targets"][-1] == "continuous_sac" else "Interval\nrobust"
    for ax, area, label in zip(axes.flat, (1, 2), ("Mixed response", "Cross-architecture specific"), strict=True):
        ax.text(0.5, 1.04, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        for method in methods:
            _line(ax, np.arange(4), values[:, budget["methods"].index(method), area], method, marker="o", markersize=2.7)
        ax.set_xticks(range(4), ("OnlineMC", "Attention", "Transformer", last_label))
        ax.set_ylim(0, max(1, values[:, :, 1:].max() * 1.15))
    axes[0, 0].set_ylabel("Collisions found")
    _legend(fig, axes[0, 0], 3)
    _save(fig, axes, "cross_architecture_regions")


def prediction(report):
    data = report["prediction"]
    checkpoints = data["checkpoints"]
    metrics = data["metrics"]
    methods = data["methods"]
    values = np.asarray(data["values"], dtype=float)
    full, target = (methods.index(m) for m in ("lifelong_full", "target_only"))
    display = {"nll": "NLL ↓", "brier": "Brier score ↓", "recall_100": "Recall@100 ↑",
               "auroc": "AUROC ↑", "auprc_ap": "Average precision ↑", "ece": "ECE ↓"}
    comparisons = [(metrics.index(name), 1 if name in ("nll", "brier", "ece") else -1) for name in display]
    fig, axes = _panels(3, 2, height=6.2, bottom=0.12)
    fig.subplots_adjust(left=0.20, right=0.90, wspace=0.88)
    for ax, (metric, sign) in zip(axes.flat, comparisons, strict=True):
        delta = sign * finite_mean(values[:, :, target, :, metric] - values[:, :, full, :, metric], axis=1)
        finite = np.abs(delta[np.isfinite(delta)])
        limit = max(1e-6, finite.max()) if len(finite) else 1.0
        image = ax.imshow(np.ma.masked_invalid(delta), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(checkpoints)), checkpoints)
        ax.set_yticks(range(12), [LABELS[t] for t in report["targets"]], fontsize=5.6)
        ax.tick_params(length=0)
        if ax in axes[-1]:
            ax.set_xlabel("Paid feedback")
        ax.set_title(display[metrics[metric]], fontsize=8)
        box = ax.get_position()
        color_axis = fig.add_axes((box.x1 + 0.012, box.y0, 0.012, box.height))
        fig.colorbar(image, cax=color_axis)
    _save(fig, axes, "prediction_gain_heatmaps")


def reliability(report):
    data = report["prediction"]
    values = np.asarray(data["values"], dtype=float)
    historical = np.asarray(data["historical_weight"])
    methods = data["methods"]
    checkpoints = data["checkpoints"]
    fig, axes = _panels(2, 2, height=4.6, bottom=0.17, sharex=True, sharey="row")
    for col, (block, name) in enumerate(((slice(0, 8), "In-family"), (slice(8, 12), "Cross-architecture"))):
        ax, weights = axes[:, col]
        ax.set_title(name, fontsize=8)
        for method in ("target_only", "full", "frozen", "lifelong_full"):
            line = values[block, :, methods.index(method), :, 0].mean(axis=(0, 1))
            _line(ax, range(len(checkpoints)), line, method, marker="o", markersize=2.5)
        weights.plot(range(len(checkpoints)), historical[block].mean(axis=(0, 1)),
                     color=COLORS["lifelong_full"], linewidth=1.5, marker="o", markersize=2.5)
        weights.set_ylim(0, 1)
        weights.set_xticks(range(len(checkpoints)), checkpoints)
        weights.set_xlabel("Paid feedback")
    axes[0, 0].set_ylabel("Held-out NLL ↓")
    axes[1, 0].set_ylabel("Historical weight")
    _legend(fig, axes[0, 0], 4)
    _save(fig, axes, "calibration_and_weights")


def continual(report):
    effects = report["continual"]["studies"]["cold_start"]["comparisons"]
    fig, axes = _panels(2, 3, height=4.25, bottom=0.22)
    fig.subplots_adjust(left=0.13, right=0.98, wspace=0.58, hspace=0.53)
    metrics = (("k0_nll", "Δ Zero-shot NLL ↓"), ("k0_recall_50", "Δ Recall@50 ↑"),
               ("nll_init", "Δ Initial NLL ↓"), ("c50", "Δ C(50) ↑"),
               ("auc100", "Δ AUC@100 ↑"), ("auc", "Δ AUC@200 ↑"))
    steps = (
        ("naive_append_minus_fixed", "Naive − fixed", "#8A633E", "--", "v"),
        ("debiased_append_minus_naive_append", "Debiased − naive", "#487C80", "-.", "s"),
        ("full_lifelong_minus_debiased_append", "Ours method − debiased", "#8A75A5", ":", "D"),
        ("full_lifelong_minus_fixed", "Ours method − fixed", "#0B5368", "-", "o"),
    )
    orders = np.arange(2, 5)
    for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
        ax.axhline(0, color="#A9A9A9", linewidth=0.7)
        for i, (contrast, legend, color, linestyle, marker) in enumerate(steps):
            rows = [effects[contrast]["by_order"][str(order)][metric] for order in orders]
            mean = np.asarray([row["mean"] for row in rows], dtype=float)
            low = np.asarray([row["ci_low"] for row in rows], dtype=float)
            high = np.asarray([row["ci_high"] for row in rows], dtype=float)
            error = np.maximum(0, np.vstack((mean - low, high - mean)))
            x = orders + (i - 1.5) * 0.09
            ax.errorbar(x, mean, yerr=error, color=color, linestyle=linestyle, marker=marker,
                        markersize=3, linewidth=1.3 if i == 3 else 0.8, capsize=1.6, label=legend)
        ax.set_xticks(orders)
        ax.set_ylabel(label)
    for ax in axes[-1]:
        ax.set_xlabel("Task in sequence")
    _legend(fig, axes[0, 0], 2)
    _save(fig, axes, "cold_start_growth")


def local_reliability(report):
    case = report["driving_cases"]["cases"]["online_mc_ppo"]
    surface = case["common_prefix_surface"]
    fields = {
        "car_following_braking_onset": (("gap_m", "Initial clearance (m)"), ("closing_speed_mps", "Closing speed (m/s)")),
        "cut_in_onset": (("gap_m", "Initial clearance (m)"), ("closing_speed_mps", "Closing speed (m/s)")),
        "cut_out_rear_gap": (("obstacle_distance_m", "Initial obstacle distance (m)"), ("rear_gap_m", "Rear gap (m)")),
        "side_encroachment": (("longitudinal_offset_m", "Intruder offset (m)"), ("relative_speed_mps", "Relative speed (m/s)")),
    }
    (x_key, x_label), (y_key, y_label) = fields[case["scenario"]]
    coords = np.asarray([[point[x_key], point[y_key]] for point in surface["parameters"]])
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    change = np.asarray(surface["local_probabilities"]) - surface["v1_probabilities"]
    weights = np.asarray(surface["local_historical_weight"])
    limit = max(0.001, float(np.abs(change).max()))
    weight_limits = (np.floor(20 * weights.min()) / 20, np.ceil(20 * weights.max()) / 20)
    fig, axes = _panels(1, 2, height=3.15, bottom=0.23)
    fig.subplots_adjust(left=0.13, right=0.90, wspace=0.65, top=0.84)
    for ax, (values, cmap, low, high, label) in zip(axes.flat, (
        (change, "RdBu", -limit, limit, "Δ Collision probability"),
        (weights, "viridis", *weight_limits, "Historical weight"),
    ), strict=True):
        means = np.bincount(inverse, weights=values) / np.bincount(inverse)
        image = ax.scatter(*unique.T, c=means, s=12, marker="s", cmap=cmap, vmin=low, vmax=high, linewidths=0)
        ax.scatter(*coords[case["candidate_index"]], marker="*", s=64, facecolor="white",
                   edgecolor="black", linewidth=0.7, label="Collision found only by +LocalRel", zorder=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        box = ax.get_position()
        color_axis = fig.add_axes((box.x1 + 0.012, box.y0, 0.012, box.height))
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label(label)
    _legend(fig, axes[0, 0], 1)
    _save(fig, axes, "local_reliability_surface")


def main():
    report = read_json(REPORT / "summary.json")
    verify_report_links()
    with plt.rc_context(SMALL):
        for draw in (discovery, composition, regions, prediction, reliability, continual, local_reliability):
            draw(report)


if __name__ == "__main__":
    main()
