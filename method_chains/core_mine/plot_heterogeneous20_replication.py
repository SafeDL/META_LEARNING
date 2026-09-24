"""Figures for the frozen heterogeneous 20 Hz physical replication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method_chains.core_mine.heterogeneous20_replication import METHODS, ROOT, SEEDS


COLORS = {
    "HistoryMargin-Residual": "#176C8A",
    "CoRe-Residual": "#6C4C9C",
    "HistoryMargin-Static": "#D17A2C",
    "ModeQuantile-Static": "#C2A136",
    "TargetOnly-Residual": "#478B56",
    "RandomSafe": "#777777",
}


def main() -> None:
    analysis_path = ROOT / "analysis50.json"
    if not analysis_path.exists():
        raise RuntimeError("run and verify all physical campaigns before plotting")
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    if data["total_target_physical_episodes"] != 900:
        raise RuntimeError("incomplete physical replication")
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    methods = [item[0] for item in METHODS]
    by_key = {(item["seed"], item["method"]): item for item in data["unit_rows"]}

    fig, axis = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(SEEDS))
    width = .13
    for order, method in enumerate(methods):
        values = [by_key[(seed, method)]["new_failures"] for seed in SEEDS]
        axis.bar(positions + (order - 2.5) * width, values, width,
                 color=COLORS[method], label=method)
    axis.set_xticks(positions, [str(seed) for seed in SEEDS])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Prospective simulator seed")
    axis.set_ylabel("New target failures in 50 physical tests")
    axis.set_title("Previously safe scenarios that fail on VI/TTC")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, fontsize=8, frameon=False, loc="upper center",
                bbox_to_anchor=(.5, -.17))
    fig.tight_layout()
    fig.savefig(output / "failures_by_seed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(1, 51)
    for method in methods:
        curves = []
        for seed in SEEDS:
            with (ROOT / str(seed) / f"{method}.csv").open(
                    encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 50:
                raise RuntimeError(f"incomplete trace seed={seed} method={method}")
            curves.append(np.cumsum([row["event"] == "True" for row in rows]))
        values = np.mean(curves, axis=0)
        axis.plot(x, values, color=COLORS[method], label=method,
                  linewidth=2.3 if method == "HistoryMargin-Residual" else 1.7)
    axis.set_xlim(1, 50)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Charged VI/TTC test queries")
    axis.set_ylabel("Mean cumulative new failures")
    axis.set_title("Discovery trajectory, three prospective seeds")
    axis.grid(alpha=.25)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "discovery_curve.png", dpi=180)
    plt.close(fig)
    print(f"wrote figures to {output}")


if __name__ == "__main__":
    main()
