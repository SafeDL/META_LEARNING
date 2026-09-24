"""Frozen two-variant B=50 development of mode-calibrated history margins."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.idm_revision_validation import _metrics, _task, _write_csv
from method_chains.core_mine.metrics import record_metrics
from method_chains.core_mine.oracle import CacheOracle
from method_chains.core_mine.source_safe_development import campaign


ROOT = Path("results/method_chains/core_mine/studies/boundary_calibration_development")
DEVELOPMENT_SEEDS = (20291230, 20300315)
TARGETS = ("idm_delay07", "idm_brake3", "idm_delay07_brake3")
METHODS = ("MarginModeCal", "MarginModeRate", "HistoryMargin-Static",
           "HistoryMargin-Residual", "TargetOnly-Residual",
           "ModeQuantile-Static", "RandomSafe")


def _source_scores(task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All normalization is fitted to historical source responses only."""
    source = task.source_y.mean(axis=0)
    scale = max(float(np.std(source)), 1e-9)
    z = (source - np.median(source)) / scale
    percentile = np.empty(task.count)
    labels, mode_index = np.unique(task.modes.astype(str), return_inverse=True)
    for index in range(len(labels)):
        positions = np.flatnonzero(mode_index == index)
        percentile[positions] = (rankdata(source[positions], method="average") - 1) / (
            len(positions) - 1)
    return z, percentile, mode_index


def _mode_intercepts(z: np.ndarray, mode_index: np.ndarray,
                     selected: list[int], events: list[bool], count: int) -> np.ndarray:
    """Independent one-dimensional logistic MAP fits with fixed N(0,1.5²) priors."""
    offsets = np.zeros(count)
    if not selected:
        return offsets
    selected_array = np.asarray(selected, dtype=int)
    observed = np.asarray(events, dtype=float)
    for mode in range(count):
        same = mode_index[selected_array] == mode
        if not np.any(same):
            continue
        logits = 2.0 * z[selected_array[same]]
        labels = observed[same]
        value = 0.0
        for _ in range(20):
            probability = expit(logits + value)
            gradient = float(np.sum(probability - labels) + value / 1.5**2)
            hessian = float(np.sum(probability * (1 - probability)) + 1 / 1.5**2)
            update = gradient / hessian
            value -= update
            if abs(update) < 1e-10:
                break
        offsets[mode] = value
    return offsets


def _candidate_scores(method: str, z: np.ndarray, percentile: np.ndarray,
                      mode_index: np.ndarray, selected: list[int],
                      events: list[bool]) -> np.ndarray:
    if method == "MarginModeCal":
        offsets = _mode_intercepts(z, mode_index, selected, events,
                                   int(mode_index.max()) + 1)
        return expit(2.0 * z + offsets[mode_index])
    if method == "ModeQuantile-Static":
        return percentile.copy()
    if method == "MarginModeRate":
        rates = np.full(int(mode_index.max()) + 1, .5)
        for mode in range(len(rates)):
            picked = [flag for index, flag in zip(selected, events, strict=True)
                      if mode_index[index] == mode]
            rates[mode] = (1 + sum(picked)) / (2 + len(picked))
        return .75 * percentile + .25 * rates[mode_index]
    raise ValueError(method)


def _run(task, method: str) -> dict:
    z, percentile, mode_index = _source_scores(task)
    oracle = CacheOracle(task)
    selected: list[int] = []
    events: list[bool] = []
    while len(selected) < 50:
        scores = _candidate_scores(method, z, percentile, mode_index,
                                   selected, events)
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET)
        outcome = oracle.reveal(index)
        selected.append(index)
        events.append(bool(outcome.event))
    return record_metrics(task, selected, 50)


def main() -> None:
    rows = []
    for seed in DEVELOPMENT_SEEDS:
        for target in TARGETS:
            task = _task(seed, target)
            for method in METHODS[:-1]:
                if method in {"MarginModeCal", "MarginModeRate",
                              "ModeQuantile-Static"}:
                    record = _run(task, method)
                else:
                    branch = "target" if method == "TargetOnly-Residual" else "mean"
                    record = campaign(task, branch, 0.0,
                                      residual=method != "HistoryMargin-Static")
                rows.append(_metrics(task, {"method": method, "repeat": 0, **record}))
            for repeat in range(10):
                record = campaign(task, None, 0.0, repeat=repeat)
                rows.append(_metrics(task, {"method": "RandomSafe", "repeat": repeat,
                                            **record}))
            print(f"completed seed={seed} target={target}", flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(ROOT / "records.csv", rows)
    unit_rows = []
    for seed in DEVELOPMENT_SEEDS:
        for target in TARGETS:
            for method in METHODS:
                subset = [row for row in rows if row["seed"] == seed
                          and row["heterogeneity"] == target and row["method"] == method]
                unit_rows.append({"seed": seed, "target": target, "method": method,
                                  "pool_new_failures": subset[0]["pool_new_failures"],
                                  "new_failures": float(np.mean([row["NewHistoricalFailureCount"]
                                                                 for row in subset])),
                                  "ego_collisions": float(np.mean([row["CollisionCount"]
                                                                   for row in subset])),
                                  "cvs": float(np.mean([row["CVS"] for row in subset]))})
    by_seed = {seed: {method: {key: float(np.mean([row[key] for row in unit_rows
                  if row["seed"] == seed and row["method"] == method]))
                  for key in ("new_failures", "ego_collisions", "cvs")}
                  for method in METHODS} for seed in DEVELOPMENT_SEEDS}
    summary = {method: {key: float(np.mean([row[key] for row in unit_rows
                if row["method"] == method]))
                for key in ("new_failures", "ego_collisions", "cvs")}
                for method in METHODS}
    strongest_static = max(("HistoryMargin-Static", "ModeQuantile-Static"),
                           key=lambda method: summary[method]["new_failures"])
    qualified = {method: all(by_seed[seed][method]["new_failures"] >
                             max(by_seed[seed][baseline]["new_failures"] for baseline in
                                 ("HistoryMargin-Static", "ModeQuantile-Static"))
                             for seed in DEVELOPMENT_SEEDS)
                 and summary[method]["ego_collisions"] >= .9 *
                 summary[strongest_static]["ego_collisions"]
                 for method in ("MarginModeCal", "MarginModeRate")}
    output = {"budget": 50, "development_seeds": DEVELOPMENT_SEEDS,
              "target_builds": TARGETS, "strongest_static": strongest_static,
              "qualified": qualified, "summary": summary, "by_seed": by_seed,
              "unit_rows": unit_rows}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"summary": summary, "by_seed": by_seed,
                      "qualified": qualified}, indent=2), flush=True)


if __name__ == "__main__":
    main()
