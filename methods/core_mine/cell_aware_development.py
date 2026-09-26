"""Development-only B=50 test of a fixed collision-cell penalty."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine.acquisition import choose
from methods.core_mine.config import SUPPORT_BUDGET
from methods.core_mine.data import response_value
from methods.core_mine.heterogeneous20_replication import _csv
from methods.core_mine.multimode20_experiment import (
    BOUNDS, ROOT as FORMAL_ROOT, SEEDS, TARGETS, _cell, _cluster_pair,
    _read, _unit, gate_all, source_task, target_episode,
)
from methods.core_mine.simple_residual_ablation import corrected_scores


ROOT = Path("results/method_chains/core_mine/studies/cell_aware_development")
METHOD = "CellAware-ModeShift"
COMPARATOR = "ModeShift-Risk"
PENALTY = 0.25


def cell_aware_scores(task, selected: list[int], responses: list[float],
                      severities: list[float]) -> np.ndarray:
    """Penalize only cells with a *revealed*, charged target hazard."""
    if len(selected) != len(responses) or len(selected) != len(severities):
        raise ValueError("target-feedback ledgers must have equal lengths")
    scores = corrected_scores(task, selected, responses,
                              "HistoryMargin-ModeShift")
    occupied: dict[tuple[str, int, int], float] = {}
    for index, severity in zip(selected, severities, strict=True):
        key = _cell(str(task.modes[index]),
                    float(task.anchors[index, 0]),
                    float(task.anchors[index, 1]))
        occupied[key] = max(occupied.get(key, 0.0), severity)
    if occupied:
        for index in range(task.count):
            key = _cell(str(task.modes[index]),
                        float(task.anchors[index, 0]),
                        float(task.anchors[index, 1]))
            scores[index] -= PENALTY * occupied.get(key, 0.0)
    return scores


def run_campaign(seed: int, target: str) -> Path:
    gate_all()
    if seed not in SEEDS or target not in TARGETS:
        raise ValueError((seed, target))
    path = ROOT / str(seed) / target / f"{METHOD}.csv"
    if path.exists():
        print(f"reused {seed}/{target}/{METHOD}", flush=True)
        return path
    task, eligible = source_task(seed, target)
    selected: list[int] = []
    responses: list[float] = []
    severities: list[float] = []
    rows: list[dict] = []
    for query in range(1, 51):
        started = time.perf_counter()
        scores = cell_aware_scores(task, selected, responses, severities)
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        started = time.perf_counter()
        result = target_episode(target, seed, index, task)
        elapsed_seconds = time.perf_counter() - started
        collision = bool(result["ego_collision"])
        near_miss = bool(result["near_miss"])
        event = collision or near_miss
        response = float(response_value(
            np.asarray([result["min_ttc"]]), np.asarray([event]),
            np.asarray([collision]))[0])
        selected.append(index)
        responses.append(response)
        severities.append(1.0 if collision else .5 if near_miss else 0.0)
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
    return path


def _grid_cell(task, index: int, bins: int) -> tuple[str, int, int]:
    mode = str(task.modes[index])
    values = task.anchors[index]
    cells = []
    for value, (low, high) in zip(values, BOUNDS[mode], strict=True):
        if value < low:
            cells.append(-1)
        elif value > high:
            cells.append(bins)
        else:
            cells.append(min(bins, int(np.floor(
                bins * (value - low) / (high - low)))))
    return mode, cells[0], cells[1]


def _cell_count(task, rows: list[dict], bins: int) -> int:
    return len({_grid_cell(task, int(row["index"]), bins)
                for row in rows if row["ego_collision"] == "True"})


def analyze() -> dict:
    gate_all()
    units = []
    repeats = 0
    for seed in SEEDS:
        for target in TARGETS:
            task, eligible = source_task(seed, target)
            new_rows = _read(ROOT / str(seed) / target / f"{METHOD}.csv")
            old_rows = _read(FORMAL_ROOT / str(seed) / target
                             / f"{COMPARATOR}.csv")
            if (len(new_rows) != 50 or len(old_rows) != 50
                    or [int(row["query"]) for row in new_rows]
                    != list(range(1, 51))
                    or len({int(row["index"]) for row in new_rows}) != 50
                    or set(int(row["index"]) for row in new_rows)
                    - set(eligible.tolist())
                    or any(row["method"] != METHOD or row["target"] != target
                           for row in new_rows)):
                raise RuntimeError(f"invalid charged B=50 trace {seed}/{target}")
            old_by_index = {int(row["index"]): row for row in old_rows}
            for row in new_rows:
                if (row["event"] == "True") != (
                        row["ego_collision"] == "True"
                        or row["near_miss"] == "True"):
                    raise RuntimeError("event response contract changed")
                old = old_by_index.get(int(row["index"]))
                if old is None:
                    continue
                fields = ("ego_collision", "near_miss", "background_collision",
                          "completed", "min_ttc", "min_clearance")
                if any(row[field] != old[field] for field in fields):
                    raise RuntimeError("repeated physical scenario disagrees")
                repeats += 1
            for method, rows in ((METHOD, new_rows), (COMPARATOR, old_rows)):
                unit = _unit(task, seed, target, method, rows)
                unit["collision_cells_3x3"] = _cell_count(task, rows, 3)
                unit["collision_cells_5x5"] = _cell_count(task, rows, 5)
                units.append(unit)
    keyed = {(item["seed"], item["target"], item["method"]): item
             for item in units}
    metrics = ("collision_cells", "collision_cells_3x3",
               "collision_cells_5x5", "collision_modes",
               "ego_collisions", "new_failures", "cvs", "early_auc")
    summary = {method: {target: {metric: float(np.mean([
        keyed[(seed, target, method)][metric] for seed in SEEDS]))
        for metric in metrics} for target in TARGETS}
        for method in (METHOD, COMPARATOR)}
    for method in (METHOD, COMPARATOR):
        summary[method]["overall"] = {metric: float(np.mean([
            keyed[(seed, target, method)][metric]
            for seed in SEEDS for target in TARGETS])) for metric in metrics}
    rng = np.random.default_rng(20330620)
    paired = {metric: _cluster_pair(np.asarray([[
        keyed[(seed, target, METHOD)][metric]
        - keyed[(seed, target, COMPARATOR)][metric]
        for target in TARGETS] for seed in SEEDS], dtype=float), rng)
        for metric in metrics}
    passed = bool(
        all(paired["collision_cells"]["by_target_mean"][target] > 0
            for target in TARGETS)
        and paired["collision_cells"]["mean"] >= 1.0
        and summary[METHOD]["overall"]["ego_collisions"]
        >= .9 * summary[COMPARATOR]["overall"]["ego_collisions"])
    output = {"schema": "cell_aware_mode_shift_development_v1",
              "budget": 50, "seeds": SEEDS, "targets": TARGETS,
              "penalty": PENALTY,
              "target_outcomes_precomputed": False,
              "new_physical_target_episodes": len(SEEDS) * len(TARGETS) * 50,
              "repeated_target_scenarios_verified": repeats,
              "development_gate_passed": passed,
              "summary": summary, "paired": paired,
              "unit_rows": units}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis50.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": passed, "overall": {
        method: summary[method]["overall"]
        for method in (METHOD, COMPARATOR)},
        "primary_paired": paired["collision_cells"],
        "repeated_target_scenarios_verified": repeats}, indent=2),
        flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "analyze", "all"),
                        default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in {"targets", "all"}:
        jobs = [(seed, target) for seed in SEEDS for target in TARGETS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda_job, jobs))
    if args.stage in {"analyze", "all"}:
        analyze()


def lambda_job(args: tuple[int, str]) -> str:
    """Top-level worker so Windows process spawning can pickle the call."""
    return str(run_campaign(*args))


if __name__ == "__main__":
    main()
