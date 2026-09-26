"""Charged B=50 source-mean static control for online mode allocation."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine import multimode20_experiment as base
from methods.core_mine import mode_shift_fresh_confirmation as fresh
from methods.core_mine.acquisition import choose
from methods.core_mine.cell_aware_development import _cell_count
from methods.core_mine.config import SUPPORT_BUDGET
from methods.core_mine.data import response_value
from methods.core_mine.heterogeneous20_replication import _csv
from methods.core_mine.mode_label_ablation import (
    METHOD as LABEL_METHOD,
    ROOT as LABEL_ROOT,
    _configure,
    _outcome,
    _validate,
)


ROOT = Path("results/method_chains/core_mine/studies/source_raw_static_development")
METHOD = "SourceMeanRaw-Static"
COMPARATORS = (LABEL_METHOD, "ModeShift-Risk", "ModeQuantile-Static",
               "SourceStatic-Marginal")
METRICS = ("collision_cells", "collision_cells_3x3",
           "collision_cells_5x5", "ego_collisions", "new_failures")


def run_campaign(seed: int, target: str) -> str:
    _configure()
    base.gate_all()
    if target not in fresh.TARGETS:
        raise ValueError(target)
    path = ROOT / str(seed) / target / f"{METHOD}.csv"
    if path.exists():
        print(f"reused {seed}/{target}/{METHOD}", flush=True)
        return str(path)
    task, eligible = base.source_task(seed, target)
    scores = task.source_y.mean(axis=0)
    selected: list[int] = []
    rows: list[dict] = []
    for query in range(1, 51):
        selection_started = time.perf_counter()
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - selection_started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        episode_started = time.perf_counter()
        result = base.target_episode(target, seed, index, task)
        elapsed_seconds = time.perf_counter() - episode_started
        collision = bool(result["ego_collision"])
        near_miss = bool(result["near_miss"])
        event = collision or near_miss
        response = float(response_value(
            np.asarray([result["min_ttc"]]), np.asarray([event]),
            np.asarray([collision]))[0])
        selected.append(index)
        rows.append({"seed": seed, "target": target, "method": METHOD,
                     "query": query, "index": index,
                     "mode": str(task.modes[index]),
                     "ego_collision": collision, "near_miss": near_miss,
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response,
                     "selection_seconds": selection_seconds,
                     "elapsed_seconds": elapsed_seconds})
    path.parent.mkdir(parents=True, exist_ok=True)
    _csv(path, rows)
    print(f"executed {seed}/{target}/{METHOD}: 50", flush=True)
    return str(path)


def _target_job(job: tuple[int, str]) -> str:
    return run_campaign(*job)


def run_targets(workers: int) -> None:
    _configure()
    base.gate_all()
    jobs = [(seed, target) for seed in fresh.SEEDS for target in fresh.TARGETS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_target_job, jobs))


def verify_static_decisions(task, eligible: np.ndarray,
                            rows: list[dict]) -> None:
    """Reconstruct the entire no-feedback decision sequence."""
    scores = task.source_y.mean(axis=0)
    seen: list[int] = []
    for row in rows:
        predicted = choose(scores, seen, task.modes, SUPPORT_BUDGET,
                           allowed_indices=eligible)
        index = int(row["index"])
        if predicted != index:
            raise RuntimeError("raw static ledger disagrees with source scores")
        seen.append(index)


def analyze() -> dict:
    _configure()
    base.gate_all()
    methods = (METHOD, *COMPARATORS)
    units: list[dict] = []
    repeats: dict[tuple[int, str, int], list[tuple]] = {}
    selections: dict[tuple[int, str, str], list[int]] = {}
    for seed in fresh.SEEDS:
        for target in fresh.TARGETS:
            task, eligible = base.source_task(seed, target)
            allowed = set(eligible.tolist())
            for method in methods:
                root = (ROOT if method == METHOD else LABEL_ROOT
                        if method == LABEL_METHOD else fresh.ROOT)
                rows = base._read(root / str(seed) / target / f"{method}.csv")
                _validate(rows, seed, target, method, allowed)
                if method == METHOD:
                    verify_static_decisions(task, eligible, rows)
                selections[(seed, target, method)] = [
                    int(row["index"]) for row in rows]
                for row in rows:
                    repeats.setdefault((seed, target, int(row["index"])),
                                       []).append(_outcome(row))
                unit = base._unit(task, seed, target, method, rows)
                unit["collision_cells_3x3"] = _cell_count(task, rows, 3)
                unit["collision_cells_5x5"] = _cell_count(task, rows, 5)
                units.append(unit)
    if any(len(set(values)) != 1 for values in repeats.values()):
        raise RuntimeError("repeated physical target outcomes disagree")
    keyed = {(unit["seed"], unit["target"], unit["method"]): unit
             for unit in units}
    summary = {method: {target: {metric: float(np.mean([
        keyed[(seed, target, method)][metric] for seed in fresh.SEEDS]))
        for metric in METRICS} for target in fresh.TARGETS}
        for method in methods}
    for method in methods:
        summary[method]["overall"] = {metric: float(np.mean([
            keyed[(seed, target, method)][metric]
            for seed in fresh.SEEDS for target in fresh.TARGETS]))
            for metric in METRICS}
    matrix = np.asarray([[
        keyed[(seed, target, LABEL_METHOD)]["collision_cells"]
        - keyed[(seed, target, METHOD)]["collision_cells"]
        for target in fresh.TARGETS] for seed in fresh.SEEDS], dtype=float)
    paired = base._cluster_pair(matrix, np.random.default_rng(20330910))
    passed = bool(
        all(paired["by_target_mean"][target] > 0
            for target in fresh.TARGETS)
        and paired["seed_cluster_bootstrap_95"][0] > 0
        and summary[LABEL_METHOD]["overall"]["ego_collisions"]
        >= summary[METHOD]["overall"]["ego_collisions"])
    overlap = [{"seed": seed, "target": target,
                "shared_selected_cases": len(
                    set(selections[(seed, target, METHOD)]) &
                    set(selections[(seed, target, LABEL_METHOD)]))}
               for seed in fresh.SEEDS for target in fresh.TARGETS]
    output = {"schema": "source_mean_raw_static_development_v1",
              "development_only": True, "budget": 50,
              "seeds": fresh.SEEDS, "targets": fresh.TARGETS,
              "new_physical_target_episodes": len(fresh.SEEDS)
                                               * len(fresh.TARGETS) * 50,
              "online_allocation_development_gate": passed,
              "repeated_target_scenarios_verified": sum(
                  len(values) > 1 for values in repeats.values()),
              "label_minus_raw_primary": paired,
              "label_vs_raw_overlap": overlap,
              "summary": summary, "unit_rows": units}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"gate": passed, "summary": summary,
                      "label_minus_raw_primary": paired,
                      "overlap": overlap}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "analyze", "all"),
                        default="targets")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in ("targets", "all"):
        run_targets(args.workers)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
