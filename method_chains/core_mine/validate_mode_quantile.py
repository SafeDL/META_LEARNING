"""Frozen B=50 independent validation of source-safe mode-quantile ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from method_chains.core_mine.development_mode_calibration import _run
from method_chains.core_mine.idm_revision_validation import (
    ROOT as BANK_ROOT, TARGETS, _diverse_campaign, _metrics, _task, _write_csv,
    build_source, build_targets,
)
from method_chains.core_mine.source_safe_development import campaign


ROOT = Path("results/method_chains/core_mine/studies/mode_quantile_validation")
VALIDATION_SEEDS = (20300701, 20300715, 20300729)
METHODS = ("ModeQuantile-Static", "HistoryMargin-Static",
           "RiskDiverse-Static", "HistoryMargin-Residual",
           "TargetOnly-Residual", "RandomSafe")


def _check_gates() -> None:
    for seed in VALIDATION_SEEDS:
        path = BANK_ROOT / str(seed) / "gate.json"
        if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError(f"source-only gate missing or failed at seed {seed}")


def build_sources(workers: int) -> None:
    gates = [build_source(seed, workers) for seed in VALIDATION_SEEDS]
    if not all(gate["passed"] for gate in gates):
        raise RuntimeError("predeclared source-only gate failed; no target runs allowed")


def run_targets(workers: int) -> None:
    _check_gates()
    for seed in VALIDATION_SEEDS:
        build_targets(seed, workers)


def _paired_intervals(unit: dict[tuple[int, str, str], dict]) -> dict:
    rng = np.random.default_rng(20300801)
    output = {}
    for baseline in METHODS[1:]:
        output[baseline] = {}
        for metric in ("NewHistoricalFailureCount", "CollisionCount", "CVS"):
            diff = np.asarray([[unit[(seed, target, METHODS[0])][metric]
                                - unit[(seed, target, baseline)][metric]
                                for target in TARGETS] for seed in VALIDATION_SEEDS])
            sample = np.empty(10000)
            for draw in range(len(sample)):
                seeds = rng.integers(0, len(VALIDATION_SEEDS), len(VALIDATION_SEEDS))
                targets = rng.integers(0, len(TARGETS),
                                       (len(VALIDATION_SEEDS), len(TARGETS)))
                sample[draw] = diff[seeds[:, None], targets].mean()
            output[baseline][metric] = {
                "mean_difference": float(diff.mean()),
                "bootstrap_95_low": float(np.quantile(sample, .025)),
                "bootstrap_95_high": float(np.quantile(sample, .975)),
                "paired_unit_differences": diff.tolist(),
            }
    return output


def analyze() -> dict:
    _check_gates()
    rows = []
    for seed in VALIDATION_SEEDS:
        for target in TARGETS:
            task = _task(seed, target)
            settings = (
                ("ModeQuantile-Static", lambda: _run(task, "ModeQuantile-Static")),
                ("HistoryMargin-Static", lambda: campaign(task, "mean", 0.0,
                                                           residual=False)),
                ("RiskDiverse-Static", lambda: _diverse_campaign(task)),
                ("HistoryMargin-Residual", lambda: campaign(task, "mean", 0.0,
                                                             residual=True)),
                ("TargetOnly-Residual", lambda: campaign(task, "target", 0.0,
                                                          residual=True)),
            )
            for method, run in settings:
                rows.append(_metrics(task, {"method": method, "repeat": 0, **run()}))
            for repeat in range(10):
                rows.append(_metrics(task, {"method": "RandomSafe", "repeat": repeat,
                                            **campaign(task, None, 0.0, repeat=repeat)}))
            print(f"completed seed={seed} target={target}", flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(ROOT / "records.csv", rows)
    metrics = ("NewHistoricalFailureCount", "CollisionCount", "NovelFailureModes",
               "EarlyNovelAUC", "CVS")
    unit_rows = []
    for seed in VALIDATION_SEEDS:
        for target in TARGETS:
            for method in METHODS:
                subset = [row for row in rows if row["seed"] == seed
                          and row["heterogeneity"] == target and row["method"] == method]
                unit_rows.append({"seed": seed, "target": target, "method": method,
                                  "pool_new_failures": int(subset[0]["pool_new_failures"]),
                                  "pool_ego_collisions": int(subset[0]["pool_ego_collisions"]),
                                  **{metric: float(np.mean([row[metric] for row in subset]))
                                     for metric in metrics}})
    unit = {(row["seed"], row["target"], row["method"]): row for row in unit_rows}
    summary = {method: {metric: float(np.mean([row[metric] for row in unit_rows
                if row["method"] == method])) for metric in metrics}
                for method in METHODS}
    by_target = {target: {method: {metric: float(np.mean([row[metric]
                 for row in unit_rows if row["target"] == target and row["method"] == method]))
                 for metric in metrics} for method in METHODS} for target in TARGETS}
    paired = _paired_intervals(unit)
    qualifies = all(paired[baseline]["NewHistoricalFailureCount"]["mean_difference"] > 0
                    for baseline in ("HistoryMargin-Static", "RiskDiverse-Static"))
    output = {"budget": 50, "validation_seeds": VALIDATION_SEEDS,
              "target_builds": TARGETS, "methods": METHODS,
              "predeclared_mean_gate_passed": qualifies,
              "summary": summary, "by_target": by_target,
              "paired": paired, "unit_rows": unit_rows}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": paired,
                      "predeclared_mean_gate_passed": qualifies}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "analyze", "all"),
                        default="sources")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage in ("sources", "all"):
        build_sources(args.workers)
    if args.stage in ("targets", "all"):
        run_targets(args.workers)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
