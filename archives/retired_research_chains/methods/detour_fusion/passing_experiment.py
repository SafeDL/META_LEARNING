"""Evaluate Risk Mining and DETOUR on the paper-inspired passing benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from highway_sim_env.data.generate_anchor_bank import generate_anchor_bank
from highway_sim_env.data.response_bank import ResponseBank, build_response_bank
from sut_algorithms.highway_env.idm_profiles import (
    ADATE_SOURCE_PROFILES,
    ADATE_TARGET_PROFILES,
)
from methods.detour_fusion.config import FusionExperimentConfig
from methods.detour_fusion.experiment import (
    METHOD_COLORS,
    METHOD_ORDER,
    OUTPUT_DIR as FUSION_OUTPUT,
    run_loso_multifunction,
)

OUTPUT_DIR = FUSION_OUTPUT / "rare_event_passing"
CONFIG = FusionExperimentConfig(
    num_anchors=240,
    prior_rank=2,
    support_budget=10,
    total_budget=50,
    random_support_repeats=20,
    seed=20260918,
    detour_hierarchy_weight=0.10,
)
SOURCE_NAMES = tuple(profile.name for profile in ADATE_SOURCE_PROFILES)
TARGET_NAMES = tuple(profile.name for profile in ADATE_TARGET_PROFILES)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_curve(rows: list[dict[str, object]], method: str, column: str) -> np.ndarray:
    curves = [
        np.asarray(str(row[column]).split(";"), dtype=float)
        for row in rows
        if row["method"] == method
    ]
    return np.mean(np.vstack(curves), axis=0)


def _write_curves(rows: list[dict[str, object]], output: Path) -> None:
    steps = np.arange(1, CONFIG.total_budget + 1)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for method in METHOD_ORDER:
        axis.plot(
            steps,
            _mean_curve(rows, method, "critical_recall_curve"),
            label=method,
            color=METHOD_COLORS[method],
            linewidth=2,
        )
    axis.set(
        xlabel="Test queries consumed",
        ylabel="Mean critical-event recall",
        title="Rare-event passing benchmark",
        xlim=(1, CONFIG.total_budget),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _summary(bank: ResponseBank, rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = {}
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        metrics[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "critical_score_at_50",
                "collision_count_at_50",
                "critical_event_count_at_50",
                "critical_recall_at_50",
                "collision_recall_at_50",
            )
        }
    budget_recall = {
        method: {
            f"budget_{budget}": float(
                np.mean([
                    float(row["critical_recall_curve"].split(";")[budget - 1])
                    for row in rows
                    if row["method"] == method
                ])
            )
            for budget in (10, 20, 30, 50)
        }
        for method in METHOD_ORDER
    }
    return {
        "protocol": {
            "scenario": "three-vehicle passing cut-in",
            "anchors": CONFIG.num_anchors,
            "budget": CONFIG.total_budget,
            "diagnostic_budget": CONFIG.support_budget,
            "target_truth_visibility": "support/query reveal only; offline scoring",
        },
        "event_prevalence": {
            name: {
                "collisions": int(bank.collisions[index].sum()),
                "critical_events": int(
                    (bank.collisions[index] | bank.near_misses[index]).sum()
                ),
            }
            for name in TARGET_NAMES
            for index in [bank.index_of(name)]
        },
        "mean_metrics": metrics,
        "mean_critical_recall_by_budget": budget_recall,
    }


def main() -> None:
    CONFIG.validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bank_path = OUTPUT_DIR / "response_bank.npz"
    anchors = generate_anchor_bank(CONFIG.num_anchors, CONFIG.seed)
    modes = np.full(CONFIG.num_anchors, "passing_cutin", dtype="U32")
    profiles = ADATE_SOURCE_PROFILES + ADATE_TARGET_PROFILES
    if bank_path.exists():
        bank = ResponseBank.load(bank_path)
    else:
        bank = build_response_bank(anchors, CONFIG.seed, modes, profiles)
        bank.save(bank_path)
    rows = run_loso_multifunction(bank, CONFIG, TARGET_NAMES, SOURCE_NAMES)
    _write_rows(OUTPUT_DIR / "loso_mining.csv", rows)
    summary = _summary(bank, rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_curves(rows, OUTPUT_DIR / "critical_recall_curve.png")
    print(json.dumps(summary["mean_metrics"], indent=2))


if __name__ == "__main__":
    main()
