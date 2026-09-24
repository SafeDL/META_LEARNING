"""Independent B=50 outcome audit of the frozen v5 validation campaigns."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/method_chains/core_mine/studies/corrected_geometry")
SEEDS = (20281015, 20281029, 20281112)
METRICS = ("CriticalCount", "CollisionCount", "SeveritySum", "CVS", "F", "FunctionalCoverage", "EarlyCriticalAUC")
PAIRS = (
    ("CoRe-Marginal", "MeanResidual-Marginal"),
    ("CoRe-Marginal", "FPS-Marginal"),
    ("MeanResidual-Marginal", "FPS-Marginal"),
    ("MeanResidual-Marginal", "TargetOnlyGP-Marginal"),
    ("CoRe-Risk", "MeanResidual-Risk"),
)


def _bootstrap_paired(rows: list[dict], proposed: str, baseline: str, metric: str) -> dict[str, float]:
    lookup = {(row["seed"], row["target"], row["method"]): row for row in rows}
    differences = np.asarray([[lookup[seed, target, proposed][metric] - lookup[seed, target, baseline][metric]
                               for target in sorted({row["target"] for row in rows if row["seed"] == seed})]
                              for seed in SEEDS], dtype=float)
    rng = np.random.default_rng(20260923)
    sampled = np.empty(5000)
    for draw in range(len(sampled)):
        seeds = rng.integers(0, len(SEEDS), len(SEEDS))
        units = rng.integers(0, differences.shape[1], differences.shape)
        sampled[draw] = differences[seeds[:, None], units].mean()
    return {"difference": float(differences.mean()),
            "bootstrap_low": float(np.quantile(sampled, .025)),
            "bootstrap_high": float(np.quantile(sampled, .975))}


def main() -> None:
    with (ROOT / "validate" / "records.csv").open(encoding="utf-8", newline="") as handle:
        records = [row for row in csv.DictReader(handle) if int(row["budget"]) == 50]
    banks = {}
    for seed in SEEDS:
        with np.load(ROOT / "banks" / "corrected_geometry" / f"sparse_sut_bank_{seed}.npz", allow_pickle=False) as bank:
            if str(bank["safety_metric_version"]) != "polygon_clearance_1m_v1":
                raise ValueError(f"wrong safety metric in validation bank {seed}")
            banks[seed] = {key: bank[key].copy() for key in ("sut_names", "modes", "ego_collision", "near_miss")}
    by_unit: dict[tuple[int, str, str], list[dict[str, float]]] = defaultdict(list)
    for record in records:
        seed = int(record["seed"])
        target = record["heterogeneity"]
        bank = banks[seed]
        sut_index = list(bank["sut_names"].astype(str)).index(target)
        indices = np.asarray([int(value) for value in record["queried_indices"].split(";")])
        if len(indices) != 50 or len(set(indices)) != 50:
            raise ValueError("all 50 target queries must be unique and charged")
        events = bank["ego_collision"][sut_index, indices] | bank["near_miss"][sut_index, indices]
        selected_modes = bank["modes"][indices].astype(str)
        metrics = {metric: float(record[metric]) for metric in METRICS[:5]}
        metrics["FunctionalCoverage"] = float(len(set(selected_modes[events])))
        metrics["EarlyCriticalAUC"] = float(np.cumsum(events).sum() / (50 * 51 / 2))
        by_unit[seed, target, record["method"]].append(metrics)
    rows = []
    for (seed, target, method), repeats in by_unit.items():
        rows.append({"seed": seed, "target": target, "method": method,
                     **{metric: float(np.mean([repeat[metric] for repeat in repeats])) for metric in METRICS}})
    methods = sorted({row["method"] for row in rows})
    summary = {method: {metric: float(np.mean([row[metric] for row in rows if row["method"] == method]))
                        for metric in METRICS} for method in methods}
    comparisons = {f"{proposed} vs {baseline}": {
        metric: _bootstrap_paired(rows, proposed, baseline, metric) for metric in METRICS
    } for proposed, baseline in PAIRS}
    report = {"budget": 50, "validation_seeds": SEEDS, "target_units": 9,
              "random_repetitions_per_unit": 10, "summary": summary, "paired": comparisons}
    (ROOT / "analysis50.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    display = ("MeanResidual-Marginal", "CoRe-Marginal", "FPS-Marginal", "MeanResidual-Risk",
               "CoRe-Risk", "TargetOnlyGP-Marginal", "Random")
    lines = ["# Corrected geometry validation at B=50", "", "All events below use ego collision or physical near miss.", "",
             "| Method | Critical count | Collision count | Critical modes | CVS | Early AUC |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in display:
        values = summary[method]
        lines.append(f"| {method} | {values['CriticalCount']:.2f} | {values['CollisionCount']:.2f} | "
                     f"{values['FunctionalCoverage']:.2f} | {values['CVS']:.2f} | {values['EarlyCriticalAUC']:.3f} |")
    lines += ["", "Paired differences and hierarchical bootstrap intervals are in `analysis50.json`."]
    (ROOT / "analysis50.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
