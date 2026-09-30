"""Truly sequential physical B=50 confirmation for three static selectors."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from highway_sim_env.envs.cutin_env import CutInScenario
from methods.core_mine.data import CachedTask, _features, response_value
from methods.core_mine.development_mode_calibration import _run
from methods.core_mine.idm_revision_pilot import BUILDS, _episode
from methods.core_mine.idm_revision_validation import (
    ROOT as BANK_ROOT, TARGETS, _diverse_campaign, _write_csv, build_source,
)
from methods.core_mine.source_safe_development import campaign


ROOT = Path("results/method_chains/core_mine/studies/mode_quantile_online")
SEEDS = (20300905, 20300919, 20301003)
METHODS = ("ModeQuantile-Static", "HistoryMargin-Static", "RiskDiverse-Static")


def build_sources(workers: int) -> None:
    gates = [build_source(seed, workers) for seed in SEEDS]
    if not all(gate["passed"] for gate in gates):
        raise RuntimeError("independent source-only eligibility gate failed")


def _source_task(seed: int) -> tuple[CachedTask, np.ndarray]:
    root = BANK_ROOT / str(seed)
    gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise RuntimeError("source-only gate failed")
    with np.load(root / "candidate_pool.npz", allow_pickle=False) as data:
        anchors, modes, controls, indices = (data[key].copy() for key in
                                            ("anchors", "modes", "controls",
                                             "original_indices"))
        source_ttc = data["source_min_ttc"][None, :].copy()
        source_event = (data["source_ego_collision"] | data["source_near_miss"])[None, :]
        source_collision = data["source_ego_collision"][None, :].copy()
    if source_event.any() or len(indices) != 320:
        raise RuntimeError("candidate set is not 320 historically safe scenarios")
    features, dimensions = _features(anchors, modes, controls)
    zeros = np.zeros(len(indices), dtype=bool)
    task = CachedTask(seed, "Target-source-only-static", "versioned", "source-only",
                      anchors, modes, controls, features, dimensions,
                      response_value(source_ttc, source_event, source_collision),
                      source_event, source_collision,
                      np.zeros(len(indices)), zeros, zeros,
                      np.full(len(indices), np.inf), zeros)
    return task, indices


def _orders(seed: int) -> dict[str, list[int]]:
    task, _ = _source_task(seed)
    output = {}
    for method in METHODS:
        if method == "ModeQuantile-Static":
            row = _run(task, method)
        elif method == "HistoryMargin-Static":
            row = campaign(task, "mean", 0.0, residual=False)
        else:
            row = _diverse_campaign(task)
        indices = [int(item) for item in row["queried_indices"].split(";")]
        if len(indices) != 50 or len(set(indices)) != 50:
            raise RuntimeError("a method did not select 50 distinct candidates")
        output[method] = indices
    return output


def _job(args: tuple[int, str, str, list[int]]) -> list[dict]:
    seed, target, method, order = args
    root = BANK_ROOT / str(seed)
    with np.load(root / "candidate_pool.npz", allow_pickle=False) as bank:
        originals, anchors, modes, controls = (bank[key].copy() for key in
                                              ("original_indices", "anchors",
                                               "modes", "controls"))
    rows = []
    for query, index in enumerate(order, start=1):
        scenario = CutInScenario(float(anchors[index, 0]), float(anchors[index, 1]),
                                 str(modes[index]), float(controls[index, 0]),
                                 float(controls[index, 1]))
        outcome = _episode(BUILDS[target], scenario, seed + int(originals[index]))
        rows.append({"seed": seed, "target": target, "method": method,
                     "query": query, "candidate_index": index,
                     "original_index": int(originals[index]),
                     "mode": str(modes[index]),
                     **outcome})
    return rows


def _cvs(seed: int, rows: list[dict]) -> float:
    task, _ = _source_task(seed)
    best = {}
    for row in rows:
        index = row["candidate_index"]
        cell = tuple(np.clip(np.floor(task.features[index, :2] * 4).astype(int), 0, 4))
        key = (row["mode"], *cell)
        severity = 1.0 if row["ego_collision"] else .5 if row["near_miss"] else 0.0
        best[key] = max(best.get(key, 0.0), severity)
    return float(sum(best.values()))


def run_targets(workers: int) -> dict:
    jobs = []
    for seed in SEEDS:
        for method, order in _orders(seed).items():
            for target in TARGETS:
                jobs.append((seed, target, method, order))
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        groups = list(executor.map(_job, jobs))
    rows = [row for group in groups for row in group]
    if len(rows) != len(SEEDS) * len(TARGETS) * len(METHODS) * 50:
        raise RuntimeError("physical target episode count differs from 1350")
    repeated: dict[tuple[int, str, int], tuple] = {}
    duplicate_observations = 0
    for row in rows:
        key = (row["seed"], row["target"], row["original_index"])
        value = (row["ego_collision"], row["near_miss"], row["completed"],
                 row["min_ttc"], row["min_clearance"])
        if key in repeated:
            duplicate_observations += 1
            old = repeated[key]
            if old[:3] != value[:3] or not np.allclose(old[3:], value[3:],
                                                       atol=1e-9, equal_nan=True):
                raise RuntimeError(f"cross-selector replay mismatch: {key}")
        else:
            repeated[key] = value
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(ROOT / "physical_queries.csv", rows)
    unit_rows = []
    for seed in SEEDS:
        for target in TARGETS:
            for method in METHODS:
                subset = [row for row in rows if row["seed"] == seed
                          and row["target"] == target and row["method"] == method]
                events = np.asarray([row["ego_collision"] or row["near_miss"]
                                     for row in subset], dtype=int)
                if len(subset) != 50 or len({row["candidate_index"] for row in subset}) != 50:
                    raise RuntimeError("physical campaign did not spend 50 unique queries")
                unit_rows.append({"seed": seed, "target": target, "method": method,
                                  "new_failures": int(events.sum()),
                                  "ego_collisions": int(sum(row["ego_collision"]
                                                            for row in subset)),
                                  "failure_modes": len({row["mode"] for row in subset
                                                       if row["ego_collision"] or row["near_miss"]}),
                                  "early_novel_auc": float(np.cumsum(events).sum()
                                                           / (50 * 51 / 2)),
                                  "cvs": _cvs(seed, subset)})
    summary = {method: {metric: float(np.mean([row[metric] for row in unit_rows
               if row["method"] == method]))
               for metric in ("new_failures", "ego_collisions", "failure_modes",
                              "early_novel_auc", "cvs")} for method in METHODS}
    output = {"budget": 50, "source_seeds": SEEDS,
              "source_only_gate": "320 safe candidates per seed",
              "target_outcomes_precomputed": False,
              "physical_target_episodes": len(rows),
              "cross_selector_duplicate_executions_verified": duplicate_observations,
              "summary": summary, "unit_rows": unit_rows}
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "physical_target_episodes": len(rows),
                      "cross_selector_duplicate_executions_verified":
                      duplicate_observations}, indent=2), flush=True)
    return output


def analyze_physical() -> dict:
    """Post-run paired uncertainty without executing or changing any target query."""
    data = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    unit = {(row["seed"], row["target"], row["method"]): row
            for row in data["unit_rows"]}
    rng = np.random.default_rng(20301101)
    paired = {}
    for baseline in METHODS[1:]:
        paired[baseline] = {}
        for metric in ("new_failures", "ego_collisions", "cvs"):
            diff = np.asarray([[unit[(seed, target, METHODS[0])][metric]
                                - unit[(seed, target, baseline)][metric]
                                for target in TARGETS] for seed in SEEDS])
            sample = np.empty(10000)
            for draw in range(len(sample)):
                seeds = rng.integers(0, len(SEEDS), len(SEEDS))
                targets = rng.integers(0, len(TARGETS), (len(SEEDS), len(TARGETS)))
                sample[draw] = diff[seeds[:, None], targets].mean()
            paired[baseline][metric] = {
                "mean_difference": float(diff.mean()),
                "bootstrap_95_low": float(np.quantile(sample, .025)),
                "bootstrap_95_high": float(np.quantile(sample, .975)),
                "paired_unit_differences": diff.tolist(),
            }
    output = {"budget": 50, "seeds": SEEDS, "targets": TARGETS,
              "physical_target_episodes": data["physical_target_episodes"],
              "cross_selector_duplicate_executions_verified":
              data["cross_selector_duplicate_executions_verified"],
              "summary": data["summary"], "paired": paired,
              "unit_rows": data["unit_rows"]}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"paired": paired}, indent=2), flush=True)
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
        analyze_physical()


if __name__ == "__main__":
    main()
