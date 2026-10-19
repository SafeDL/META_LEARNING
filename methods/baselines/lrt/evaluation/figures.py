"""Publication figures for the two outcomes and evaluator-only continuation audit."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.baselines.hgt.io import read_json

from ..config import ROOT


def save(figure, name):
    folder = ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(folder / f"{name}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main(comparison):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    stages = ("original_pools", "third_existing_pool")
    variants = ("cg_brt", "previous", "full")
    labels = ("CG-BRT", "DART", "LRT")
    colors = ("#4c6385", "#d1944d", "#318576")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for axis, metric, title, ceiling in zip(
        axes,
        ("H200", "high_risk_recall_pct"),
        ("Dangerous scenarios found", "Dangerous-scenario recall (%)"),
        (200, 100),
    ):
        for j, (variant, label, color) in enumerate(zip(variants, labels, colors)):
            values = [
                comparison["sections"][stage]["aggregate"][variant][metric]
                for stage in stages
            ]
            bars = axis.bar(
                np.arange(2) + (j - 1) * 0.24,
                values,
                width=0.23,
                label=label,
                color=color,
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        for i, stage in enumerate(stages):
            floor = 0.9 * comparison["sections"][stage]["aggregate"]["cg_brt"][metric]
            axis.hlines(
                floor, i - 0.38, i + 0.38, color="#333333", linestyle="--", linewidth=1
            )
        axis.set_xticks([0, 1], ["Original two pools", "Third existing pool"])
        axis.set_ylabel(title)
        axis.set_ylim(0, ceiling)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="outside lower center", ncol=3, frameon=False
    )
    save(figure, "discovery_outcomes")

    records = [
        row
        for case in read_json(ROOT / "diagnostics.json")["case_files"]
        for row in read_json(ROOT / case)["audits"]
        if row["variant"] == "full" and row["selected"] != row["reference"]
    ]
    figure, axis = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    all_values = [0.0]
    for step, color in zip(
        (10, 50, 100, 150, 190), plt.get_cmap("viridis")(np.linspace(0.08, 0.9, 5))
    ):
        rows = [row for row in records if row["observations"] == step]
        x = np.array([r["predicted_gain"] for r in rows]) / (200 - step)
        y = np.array([r["realized_gain"] for r in rows]) / (200 - step)
        all_values.extend(x.tolist() + y.tolist())
        axis.scatter(x, y, s=24, alpha=0.65, color=color, label=f"After {step} queries")
    low, high = min(all_values), max(all_values)
    margin = max(0.02, 0.08 * (high - low))
    axis.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        "--",
        color="#666666",
        linewidth=1,
    )
    axis.axhline(0, color="#bbbbbb", linewidth=0.7)
    axis.axvline(0, color="#bbbbbb", linewidth=0.7)
    axis.set(
        xlabel="Predicted continuation gain / remaining budget",
        ylabel="Realized CG-BRT branch gain / remaining budget",
    )
    axis.set_xlim(low - margin, high + margin)
    axis.set_ylim(low - margin, high + margin)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.15)
    save(figure, "continuation_audit")
