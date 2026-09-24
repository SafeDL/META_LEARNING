"""Paper-facing descriptive figures from frozen v8 and cadence-audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/method_chains/core_mine/studies/source_safe")
FIGURES = ROOT / "figures"
METHODS = ("HistoryMargin-Residual", "HistoryMargin-Static", "TargetOnly-Residual",
           "CoRe-Residual", "RandomSafe")
LABELS = ("History + residual", "History only", "Target only", "CoRe", "Random")


def plot_b50() -> None:
    analysis = json.loads((ROOT / "analysis50.json").read_text(encoding="utf-8"))
    units = analysis["unit_rows"]
    targets = ("mcts_cv", "vi_ttc")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    colors = ("#245a81", "#789ab1", "#bd9a5f", "#8071a6", "#aaaaaa")
    for axis, target in zip(axes, targets, strict=True):
        means = [np.mean([float(row["NewHistoricalFailureCount"])
                          for row in units if row["target"] == target and row["method"] == method])
                 for method in METHODS]
        axis.bar(np.arange(len(METHODS)), means, color=colors, width=.72)
        for index, value in enumerate(means):
            axis.text(index, value + .55, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
        axis.set_title(target.replace("_", " ").upper())
        axis.set_xticks(np.arange(len(METHODS)), LABELS, rotation=35, ha="right")
        axis.set_ylim(0, max(40, max(means) + 4))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("New target failures among 50 source-safe tests")
    fig.suptitle("V8 validation: method gain depends on the target controller")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_target_b50.png", dpi=200)
    plt.close(fig)


def plot_frequency() -> None:
    audit = json.loads((ROOT / "frequency_audit" / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), sharey=True)
    for axis, target in zip(axes, ("mcts_cv", "vi_ttc"), strict=True):
        at10 = audit["groups"][f"{target}_10hz"]
        at20 = audit["groups"][f"{target}_20hz"]
        total = [at10["events_5hz"], at10["events_new"], at20["events_new"]]
        collisions = [at10["collisions_5hz"], at10["collisions_new"], at20["collisions_new"]]
        x = np.arange(3)
        near_misses = np.asarray(total) - np.asarray(collisions)
        axis.bar(x, collisions, color="#bd6966", label="Ego collision")
        axis.bar(x, near_misses, bottom=collisions, color="#698fac",
                 label="Near miss (no collision)")
        for index, (event, collision) in enumerate(zip(total, collisions, strict=True)):
            axis.text(index, event + 1.3, str(event), ha="center", fontsize=9)
            axis.text(index, collision / 2, str(collision), ha="center", va="center",
                      color="white", fontsize=9)
        axis.set_xticks(x, ("5 Hz", "10 Hz", "20 Hz"))
        axis.set_title(target.replace("_", " ").upper())
        axis.set_ylim(0, 115)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Events on the same 150 selected scenarios")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Diagnostic only: fixed v8 queries rerun with faster decisions")
    fig.tight_layout()
    fig.savefig(FIGURES / "fixed_queries_cadence.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_b50()
    plot_frequency()
    print(f"wrote two evidence figures to {FIGURES}")


if __name__ == "__main__":
    main()
