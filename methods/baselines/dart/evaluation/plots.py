"""Export quantitative figures directly from the declared comparison records."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.common.catalog import DART_VARIANTS
from .config import ROOT


def plot_report(report):
    stage = report["stage"]
    aggregate = report["aggregate"]
    rows = report["rows"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    policies = list(aggregate)
    for axis, metric, label in (
        (axes[0, 0], "H200", "Discovery count retention"),
        (axes[0, 1], "high_risk_recall_pct", "Coverage retention"),
    ):
        ratios = [aggregate[p][metric] / aggregate["cg_brt"][metric] for p in policies]
        axis.barh(
            [DART_VARIANTS[p] for p in policies],
            ratios,
            color=["#777777" if p == "cg_brt" else "#2678ad" for p in policies],
        )
        axis.axvline(0.9, color="#c65b43", linestyle="--", label="Acceptance floor")
        axis.set(xlabel=label, xlim=(0, max(1.15, max(ratios) + 0.05)))
        axis.legend(fontsize=8)
    full = [r for r in rows if r["policy"] == "full"]
    predicted = [r["posterior_same_history_loss"] for r in full]
    actual = [r["actual_same_history_loss"] for r in full]
    axes[1, 0].scatter(predicted, actual, alpha=0.7, s=22)
    bound = max([1.0, *predicted, *actual])
    axes[1, 0].plot([0, bound], [0, bound], color="#777777", linestyle="--")
    axes[1, 0].set(
        xlabel="Sum of predicted replacement losses",
        ylabel="Evaluator-only realized losses",
    )
    systems = list(report["by_sut"])
    y = np.arange(len(systems))
    for offset, metric, label, color in (
        (-0.18, "H200", "Discovery count", "#2678ad"),
        (0.18, "high_risk_recall_pct", "Coverage", "#e09b40"),
    ):
        ratios = [
            report["by_sut"][s]["full"][metric] / report["by_sut"][s]["cg_brt"][metric]
            for s in systems
        ]
        axes[1, 1].barh(y + offset, ratios, height=0.32, color=color, label=label)
    axes[1, 1].axvline(0.9, color="#c65b43", linestyle="--")
    axes[1, 1].set(xlabel="Per-system retention", yticks=y, yticklabels=systems)
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(f"DART: {stage}")
    figure.savefig(ROOT / f"{stage}_results.png", dpi=200)
    figure.savefig(ROOT / f"{stage}_results.pdf")
    plt.close(figure)
