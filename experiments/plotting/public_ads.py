"""Publication figures for the admitted public-source ADS cohort (E01-E03, E05, and regression)."""

import argparse
import csv
from collections import defaultdict

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from experiments.budget import protocol as budget_protocol
from experiments.budget.protocol import sha
from experiments.paper_results import paper_study
from experiments.plotting.fixed_regression import (
    SCENES as REGRESSION_SCENES,
    dqn_query_savings,
    panel_style,
    recall,
    save as save_regression,
)
from experiments.plotting.style import STYLE, audit_figure, plt
from experiments.plotting.transfer_sequence import SCENES as SEQUENCE_SCENES
from experiments.protocol import ROOT
from experiments.public_ads.evaluate_e01 import OUTPUT as PUBLIC, TARGETS
from experiments.public_ads.extend import BUDGET, FROZEN
from experiments.regression.execution import ATTENTION
from experiments.regression.fixed_inputs import VERSIONS
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import read_json

KSHOT_RESULTS = PUBLIC / "k_shot"
KSHOT_FIGURES = ROOT / "figures/k_shot_adaptation/public_ads"
BUDGET_FIGURES = ROOT / "figures/budget_discovery"
BUDGET_PUBLIC_FIGURES = BUDGET_FIGURES / "public_ads"
WEIGHT_FIGURES = ROOT / "figures/weight_adaptation"
WEIGHT_PUBLIC_FIGURES = WEIGHT_FIGURES / "public_ads"
ACCUMULATION_FIGURES = ROOT / "figures/knowledge_accumulation/public_ads"

CHECKPOINTS = (0, 5, 10, 20, 50, 100, 200)
SCENES = ("Single-lane braking", "Restricted cut-in", "Cut-out", "Side encroachment")
KSHOT_METHODS = (
    ("target_only", "Target-only GP", "#666666", "--", "o"),
    ("pooled_history", "Pooled historical GP", "#3A78A4", "-.", "^"),
    ("full", "Ours method", "#CB3434", "-", "o"),
)
BUDGET_METHODS = {
    "full": ("Ours method", "#D04444", "-"),
    "target_only": ("Target-only GP", "#3979B5", "--"),
    "pooled_history": ("Pooled historical GP", "#C58B37", ":"),
    "scenario_fuzz": ("ScenarioFuzz", "#559A69", "-."),
    "paired_system_active_evaluation": ("Paired-system", "#9470B3", (0, (4, 1, 1, 1))),
    "random": ("Random", "#888888", (0, (1, 2))),
    "static_prior": ("Static prior", "#8F592F", (0, (5, 2))),
}
COMPACT = {
    **STYLE,
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
}


def panel(ax, letter, group=None, *, grid_axis="y"):
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", weight="bold", va="bottom")
    if group:
        ax.annotate(group, (0, 1), xycoords="axes fraction", xytext=(12, 5),
                    textcoords="offset points", va="bottom")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E5E8EB", linewidth=0.45)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def save(fig, axes, destination, name, qa_root):
    destination.mkdir(parents=True, exist_ok=True)
    audit_figure(fig, list(np.ravel(axes)), name, report_root=qa_root)
    fig.savefig(destination / f"{name}.png", dpi=600)
    plt.close(fig)


def e01_curves(targets=TARGETS):
    qa_root = PUBLIC / "reports/figure_quality"
    e01_style = {**COMPACT, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    for sut_id in targets:
        if sut_id not in TARGETS:
            raise ValueError("Only admitted public ADSs have formal E01 figures")
        with plt.rc_context(e01_style):
            fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.55), sharex=True)
            fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.17,
                                wspace=0.30, hspace=0.30)
            x = np.arange(len(CHECKPOINTS))
            for column, kind in enumerate(FORMAL_SCENARIOS):
                directory = KSHOT_RESULTS / sut_id / kind / "11"
                values = {}
                for method, *_ in KSHOT_METHODS:
                    metrics = read_json(directory / f"{method}.json.gz")["metrics"]
                    if tuple(item["K"] for item in metrics) != CHECKPOINTS:
                        raise ValueError("Public ADS curve has an incomplete fixed K sequence")
                    values[method] = metrics
                for row, (field, name) in enumerate((("nll", "NLL ↓"), ("recall_100", "Recall@100 ↑"))):
                    axis = axes[row, column]
                    axis.axvspan(0.72, 4.28, color="#F4F6F8", zorder=0)
                    for method, label, color, line, marker in KSHOT_METHODS:
                        y = [item[field] for item in values[method]]
                        if any(item is None or not np.isfinite(item) for item in y):
                            raise ValueError(f"Undefined {name} cannot be drawn as a continuous curve")
                        axis.plot(x, y, color=color, ls=line, marker=marker, ms=3.3,
                                  lw=1.4 if method == "full" else 1.1, label=label,
                                  markerfacecolor="white" if method == "target_only" else color,
                                  markeredgewidth=0.75, zorder=3 if method == "full" else 2)
                    axis.grid(axis="y", color="#E2E6EA", linewidth=0.5)
                    axis.set_axisbelow(True)
                    axis.tick_params(axis="both", length=2.5)
                    axis.annotate(chr(97 + row * 4 + column), (0, 1), xycoords="axes fraction",
                                  xytext=(0, 5), textcoords="offset points", va="bottom", ha="left",
                                  fontsize=8.5, fontweight="bold")
                    if column == 0:
                        axis.set_ylabel(name)
                    if row == 0:
                        axis.annotate(SCENES[column], (0, 1), xycoords="axes fraction",
                                      xytext=(11, 5), textcoords="offset points", va="bottom", ha="left")
                    else:
                        axis.set_xticks(x, [str(k) for k in CHECKPOINTS])
                        axis.set_xlabel("K (queries)")
                    if field == "recall_100":
                        axis.set_ylim(bottom=0)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3,
                       bbox_to_anchor=(0.5, 0.015), columnspacing=1.5, frameon=False)
            save(fig, axes, KSHOT_FIGURES, sut_id, qa_root)


def budget_curves():
    qa_root = BUDGET / "reports/figure_quality"
    receipt = read_json(BUDGET / "reports/summary.json")
    if receipt["status"] != "complete" or receipt["tasks"] != 84:
        raise ValueError("All public E02 results must pass independent audit before plotting")
    for target in TARGETS:
        with plt.rc_context(COMPACT):
            fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.85), sharex=True)
            fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.33, wspace=0.32)
            for i, (kind, group, ax) in enumerate(zip(FORMAL_SCENARIOS, SCENES, axes, strict=True)):
                panel(ax, chr(97 + i), group)
                ax.axvspan(0, 20, color="#F2F4F7", zorder=0)
                ax.axvline(20, color="#B5BCC4", ls=":", lw=0.7, zorder=1)
                for method in reversed(BUDGET_METHODS):
                    label, color, line = BUDGET_METHODS[method]
                    path = BUDGET / "budget" / target / kind / "11" / method / "result.json.gz"
                    curve = read_json(path)["metrics"]["discovery_curve"]
                    if len(curve) != 200 or any(b < a for a, b in zip(curve, curve[1:])):
                        raise ValueError("Invalid discovery curve")
                    ax.step(range(201), [0, *curve], where="post", color=color, ls=line,
                            lw=1.55 if method == "full" else 1.0, label=label,
                            zorder=4 if method == "full" else 2)
                ax.set(xlim=(0, 200), ylim=(0, None), xticks=(0, 100, 200), xlabel="Paid queries")
                if i == 0:
                    ax.set_ylabel("Collisions found ↑")
            handles = [Line2D([], [], color=c, ls=line, lw=1.55 if label == "Ours method" else 1.0, label=label)
                       for label, c, line in BUDGET_METHODS.values()]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                       ncols=4, columnspacing=1.35, frameon=False)
            save(fig, axes, BUDGET_PUBLIC_FIGURES, target, qa_root)


def regional_comparison():
    qa_root = BUDGET / "reports/figure_quality"
    primary = read_json(paper_study(budget_protocol.OUTPUT) / "reports/regional_summary.json")
    public = read_json(BUDGET / "reports/summary.json")
    if public.get("status") != "complete" or public.get("tasks") != 84:
        raise ValueError("Incomplete public E02 regional summary")
    baselines = (
        ("target_only", "vs Target-only GP", "#3979B5", "o", -0.22),
        ("pooled_history", "vs Pooled historical GP", "#C58B37", "s", 0.0),
        ("static_prior", "vs Static prior", "#8F592F", "^", 0.22),
    )
    primary_regions = (
        ("shared_collision", "Shared\n(all 8 primary)"),
        ("different_response", "Divergent\n(1–7 primary)"),
    )
    public_regions = (
        ("shared_original_eight_collision", "Shared\n(all 8 primary)"),
        ("original_eight_disagreement", "Divergent\n(1–7 primary)"),
        ("public_specific_vs_original_eight", "Public-only\n(0 of 8 primary)"),
    )
    with plt.rc_context(COMPACT):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.65), sharex=True)
        fig.subplots_adjust(left=0.175, right=0.98, top=0.88, bottom=0.23, wspace=0.52)
        ax_left, ax_right = axes
        x_bounds = (-52, 78)
        for ax in axes:
            ax.set_xlim(*x_bounds)
            ax.axvspan(0, x_bounds[1], color="#F3F7F4", zorder=0)
            ax.axvline(0, color="#6C757D", ls="--", lw=0.85, zorder=1)
            ax.grid(False)

        panel(ax_left, "a", "Primary 8 ADSs (95% CI)", grid_axis=None)
        for row, (key, _) in enumerate(primary_regions):
            scope = primary[key]["scopes"]["macro"]["ours_minus"]
            for method, _, color, marker, offset in baselines:
                y = row + offset
                gain = scope[method]["gain"][0]
                low, high = scope[method]["ci_low"][0], scope[method]["ci_high"][0]
                ax_left.hlines(y, low, high, color=color, lw=1.35, zorder=3)
                ax_left.plot(gain, y, marker, color=color, ms=4.8, zorder=4)
                anchor = high + 2.8 if gain >= 0 else low - 2.8
                ax_left.text(anchor, y, f"{gain:+.1f}", va="center",
                             ha="left" if gain >= 0 else "right",
                             fontsize=7.2, color=color, fontweight="bold")
        ax_left.set_yticks(range(len(primary_regions)), [label for _, label in primary_regions])
        ax_left.set_ylim(len(primary_regions) - 0.48, -0.52)
        ax_left.set_xlabel("Δ C(200) = Ours method − baseline ↑")

        panel(ax_right, "b", "Public 3 ADSs (mean)", grid_axis=None)
        for row, (key, _) in enumerate(public_regions):
            scope = public["regions"][key]["ours_minus"]
            for method, _, color, marker, offset in baselines:
                y = row + offset
                gain = scope[method]["gain"][0]
                ax_right.hlines(y, 0, gain, color=color, lw=1.25, alpha=0.65, zorder=3)
                ax_right.plot(gain, y, marker, color=color, ms=4.8, zorder=4)
                anchor = (max(gain, 0) + 3.2) if gain >= 0 else (min(gain, 0) - 3.2)
                ax_right.text(anchor, y, f"{gain:+.1f}", va="center",
                              ha="left" if gain >= 0 else "right",
                              fontsize=7.2, color=color, fontweight="bold")
        ax_right.set_yticks(range(len(public_regions)), [label for _, label in public_regions])
        ax_right.set_ylim(len(public_regions) - 0.48, -0.52)
        ax_right.set_xlabel("Δ C(200) = Ours method − baseline ↑")

        handles = [Line2D([], [], color=color, marker=marker, ms=4.8, lw=1.35, label=label)
                   for _, label, color, marker, _ in baselines]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.015), ncols=3,
                   columnspacing=1.5, frameon=False)
        save(fig, axes, BUDGET_FIGURES, "regional_discovery_comparison", qa_root)


def probability_effect():
    qa_root = FROZEN / "reports/figure_quality"
    with (FROZEN / "reports/paired_scores.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12:
        raise ValueError("Incomplete public E03 pairs")
    with plt.rc_context(COMPACT):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.45), sharey=True)
        fig.subplots_adjust(left=0.195, right=0.98, top=0.91, bottom=0.165, wspace=0.24, hspace=0.48)
        labels = ("OnlineMC PPO", "Attention DDQN", "Decision Trans.")
        panels = (
            (0, 0, "heldout_nll", "Held-out Δ NLL (nats) ↓", "Held-out candidates"),
            (0, 1, "heldout_brier", "Held-out Δ Brier ↓", "Held-out candidates"),
            (1, 0, "sequential_nll", "Next-query Δ NLL (nats) ↓", "Next query"),
            (1, 1, "sequential_brier", "Next-query Δ Brier ↓", "Next query"),
        )
        for idx, (r, c, metric, xlabel, group) in enumerate(panels):
            ax = axes[r, c]
            panel(ax, chr(97 + idx), group)
            ax.axvline(0, color="#6C757D", ls="--", lw=0.8, zorder=1)
            for j, (kind, scene, marker, color) in enumerate(zip(
                FORMAL_SCENARIOS, SCENES, ("o", "s", "^", "D"),
                ("#D04444", "#3979B5", "#C58B37", "#559A69"), strict=True
            )):
                values = [
                    float(next(item for item in rows if item["target"] == target and item["scenario"] == kind)[f"dynamic_{metric}"])
                    - float(next(item for item in rows if item["target"] == target and item["scenario"] == kind)[f"frozen_{metric}"])
                    for target in TARGETS
                ]
                ax.scatter(values, np.arange(3) + (j - 1.5) * 0.13, color=color, marker=marker,
                           s=24, label=scene, zorder=3)
            x_low, x_high = ax.get_xlim()
            ax.axvspan(x_low, 0, color="#F3F7F4", zorder=0)
            ax.set_xlim(x_low, max(x_high, abs(x_low) * 0.05))
            ax.set(yticks=range(3), yticklabels=labels, xlabel=xlabel)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.invert_yaxis()
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.52, 0.015),
                   ncols=4, frameon=False)
        save(fig, axes, WEIGHT_PUBLIC_FIGURES, "weight_probability_effect", qa_root)


def weight_curves():
    qa_root = FROZEN / "reports/figure_quality"
    with (FROZEN / "reports/weight_trajectories.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    examples = (
        ("online_mc_ppo", "cut_out_rear_gap", "OnlineMC PPO / Cut-out"),
        ("social_attention_ddqn", "cut_in_onset", "Attention DDQN / Cut-in"),
        ("decision_transformer", "cut_out_rear_gap", "Decision Trans. / Cut-out"),
    )
    with plt.rc_context(COMPACT):
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.29, wspace=0.22)
        for i, (ax, (target, kind, label)) in enumerate(zip(axes, examples, strict=True)):
            panel(ax, chr(97 + i), label)
            selected = [r for r in rows if r["target"] == target and r["scenario"] == kind and r["method"] == "dynamic"]
            if len(selected) != 200:
                raise ValueError("Incomplete E03 weight trajectory")
            for key, name, color, line, prior in (
                ("history_mass", "Historical sources (init 0.675)", "#666666", "-", 0.675),
                ("learned_target_mass", "Historical-feature target (0.225)", "#D04444", "--", 0.225),
                ("raw_target_mass", "Raw-feature target (0.100)", "#3979B5", ":", 0.100),
            ):
                ax.plot(range(201), [prior, *[float(r[key]) for r in selected]], label=name,
                        color=color, ls=line, lw=1.45 if key == "learned_target_mass" else 1.3)
            ax.set(xlim=(0, 200), ylim=(0, 1.03), xticks=(0, 50, 100, 150, 200), xlabel="Paid queries")
            if i == 0:
                ax.set_ylabel("Mixture weight")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncols=3, columnspacing=1.1, frameon=False)
        save(fig, axes, WEIGHT_FIGURES, "weight_trajectories", qa_root)


def extension():
    budget_curves()
    regional_comparison()
    probability_effect()
    weight_curves()


def accumulation():
    reports = PUBLIC / "knowledge_accumulation/reports"
    completion = read_json(reports / "completion.json")
    path = reports / "paired_differences.csv"
    if completion["positions"] != 72 or sha(path) != completion["files"][path.name]:
        raise ValueError("Public accumulation figure requires the full audited E05 matrix")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        values = {(r["scene"], r["order"], int(r["ordinal"])): r for r in csv.DictReader(stream)}
    append, updated, total = "#278379", "#785590", "#2F597E"
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 4.15), sharey=True)
        fig.subplots_adjust(left=0.285, right=0.976, top=0.9, bottom=0.24, wspace=0.16)
        ticks = []
        for y, (kind, order) in enumerate((s, o) for s in SEQUENCE_SCENES for o in ("forward", "reverse")):
            arrow = "→" if order == "forward" else "←"
            ticks.append(f"{SEQUENCE_SCENES[kind]}  {arrow}")
            for position, marker, offset in ((2, "o", -0.15), (3, "s", 0.15)):
                row = values[kind, order, position]
                if row["append_fixed_auc"]:
                    a.plot(float(row["append_fixed_auc"]), y + offset, marker=marker, color=append,
                           markersize=4.8, markerfacecolor="none")
            row = values[kind, order, 3]
            for field, offset, marker, color in (("update_append_auc", -0.15, "D", updated),
                                                  ("update_fixed_auc", 0.15, "^", total)):
                if row[field]:
                    b.plot(float(row[field]), y + offset, marker=marker, color=color, markersize=4.8,
                           markerfacecolor="none")
        a.set_yticks(range(len(ticks)), ticks)
        a.set_ylim(len(ticks) - 0.5, -0.5)
        for ax, letter in ((a, "a"), (b, "b")):
            ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 5),
                        textcoords="offset points", fontweight="bold", va="bottom")
            ax.axvline(0, color="#8A9399", linewidth=0.8, linestyle="--", zorder=1)
            ax.yaxis.grid(color="#EEF0F2", linewidth=0.55)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        a.set_xlabel("ΔAUC  (Append sources − Fixed history)", fontsize=8)
        b.set_xlabel("ΔAUC  (Model update)", fontsize=8)
        handles = [Line2D([], [], marker=m, linestyle="none", color=c, markerfacecolor="none", label=l)
                   for m, c, l in (("o", append, "Task 2"), ("s", append, "Task 3"),
                                   ("D", updated, "Ours method − Append sources"),
                                   ("^", total, "Ours method − Fixed history"))]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, 0.025), ncols=2,
                   fontsize=8, frameon=False)
        save(fig, [a, b], ACCUMULATION_FIGURES, "paired_accumulation_effects", reports / "figure_quality")


def regression():
    reports = PUBLIC / "policy_regression/reports"
    summary = read_json(reports / "summary.json")
    root = reports.parent
    audits = {}
    for kind, checksum in summary["scene_audits"].items():
        path = root / kind / "audit.json"
        if sha(path) != checksum:
            raise ValueError("Regression figure requires the audited attention replay curves")
        audits[kind] = read_json(path)
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    versions = []
    for kind, scene in audits.items():
        versions.extend(scene["versions"])
        for row in scene["curves"]:
            series = curves[kind][row["version"]][row["method"]]
            if row["query"] != len(series) + 1:
                raise ValueError("Regression curves are missing one or more paid queries")
            series.append(row["collisions"])
    totals = {(r["kind"], r["version"]): r["total_collisions"] for r in versions}
    recall(curves, totals, policy=ATTENTION)
    rates = {(row["kind"], row["version"]): 100 * row["collision_rate"] for row in versions}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.128, right=0.965, bottom=0.17, top=0.91, wspace=0.28, hspace=0.39)
        styles = (("#426C91", "o"), ("#B44341", "s"), ("#41826C", "^"), ("#845D93", "D"))
        for ax, (kind, label), (color, marker), letter in zip(
            axes.flat, REGRESSION_SCENES.items(), styles, "abcd", strict=True
        ):
            panel_style(ax, letter)
            ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(14, 5),
                        textcoords="offset points", va="bottom")
            ax.plot(VERSIONS, [rates.get((kind, v), float("nan")) for v in VERSIONS],
                    color=color, marker=marker, linewidth=1.35, markersize=4)
            ax.set(ylim=(0, 100), yticks=(0, 25, 50, 75, 100))
        fig.supxlabel("Social Attention DDQN version", y=0.045, fontsize=9)
        fig.supylabel("Collision rate (%) ↓", x=0.025, fontsize=9)
        save_regression(fig, axes.flat, "version_collision_rates", ATTENTION)
    dqn_query_savings(reports, ATTENTION, filename="contrasts.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        nargs="?",
        choices=("e01", "extension", "accumulation", "regression", "all"),
        default="all",
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    args = parser.parse_args()
    if args.study in ("e01", "all"):
        e01_curves(args.targets)
    if args.study in ("extension", "all"):
        extension()
    if args.study in ("accumulation", "all"):
        accumulation()
    if args.study in ("regression", "all"):
        regression()


if __name__ == "__main__":
    main()
