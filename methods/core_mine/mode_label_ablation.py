"""Physical B=50 ablation of continuous versus label-only target feedback.

This reuses inspected seeds and is development, not fresh confirmation.
"""

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


ROOT = Path("results/method_chains/core_mine/studies/mode_label_ablation")
METHOD = "ModeLabelShift-Risk"
COMPARATORS = ("ModeShift-Risk", "ModeQuantile-Static",
               "SourceStatic-Marginal")
METRICS = ("collision_cells", "collision_cells_3x3",
           "collision_cells_5x5", "ego_collisions", "new_failures")


def _configure() -> None:
    base.ROOT = fresh.ROOT
    base.SEEDS = fresh.SEEDS


def label_shift_scores(task, selected: list[int],
                       labels: list[float]) -> np.ndarray:
    """Rank with source TTC margins and only target C/E label feedback."""
    source = task.source_y.mean(axis=0)
    scores = source.copy()
    if not selected:
        return scores
    seen = np.asarray(selected, dtype=int)
    residual = np.asarray(labels, dtype=float) - source[seen]
    for mode in np.unique(task.modes[seen]):
        scores[task.modes == mode] += residual[task.modes[seen] == mode].mean()
    return scores


def run_campaign(seed: int, target: str) -> str:
    _configure()
    base.gate_all()
    if target not in base.TARGETS:
        raise ValueError(target)
    path = ROOT / str(seed) / target / f"{METHOD}.csv"
    if path.exists():
        print(f"reused {seed}/{target}/{METHOD}", flush=True)
        return str(path)
    task, eligible = base.source_task(seed, target)
    selected: list[int] = []
    labels: list[float] = []
    rows: list[dict] = []
    for query in range(1, 51):
        selection_started = time.perf_counter()
        scores = label_shift_scores(task, selected, labels)
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
        label = 0.5 * float(event) + 0.5 * float(collision)
        response = float(response_value(
            np.asarray([result["min_ttc"]]), np.asarray([event]),
            np.asarray([collision]))[0])
        selected.append(index)
        labels.append(label)
        rows.append({"seed": seed, "target": target, "method": METHOD,
                     "query": query, "index": index,
                     "mode": str(task.modes[index]),
                     "ego_collision": collision, "near_miss": near_miss,
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response, "label_response": label,
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


def _validate(rows: list[dict], seed: int, target: str, method: str,
              allowed: set[int]) -> None:
    indices = [int(row["index"]) for row in rows]
    if (len(rows) != 50 or len(set(indices)) != 50
            or set(indices) - allowed
            or [int(row["query"]) for row in rows] != list(range(1, 51))
            or any(row["method"] != method or row["target"] != target
                   or int(row["seed"]) != seed for row in rows)):
        raise RuntimeError(f"invalid B=50 ledger {seed}/{target}/{method}")
    if any((row["event"] == "True") !=
           (row["ego_collision"] == "True" or row["near_miss"] == "True")
           for row in rows):
        raise RuntimeError("event contract changed")


def _outcome(row: dict) -> tuple:
    return (row["ego_collision"], row["near_miss"],
            row["background_collision"], row["completed"],
            float(row["min_ttc"]), float(row["min_clearance"]))


def verify_label_decisions(task, eligible: np.ndarray,
                           rows: list[dict]) -> None:
    """Replay every decision using only labels revealed on prior queries."""
    seen: list[int] = []
    labels: list[float] = []
    for row in rows:
        predicted = choose(label_shift_scores(task, seen, labels), seen,
                           task.modes, SUPPORT_BUDGET,
                           allowed_indices=eligible)
        index = int(row["index"])
        if predicted != index:
            raise RuntimeError("label policy ledger is not sequential")
        event = row["event"] == "True"
        collision = row["ego_collision"] == "True"
        label = 0.5 * float(event) + 0.5 * float(collision)
        if not np.isclose(float(row["label_response"]), label):
            raise RuntimeError("stored label response disagrees with event")
        seen.append(index)
        labels.append(label)


def analyze() -> dict:
    _configure()
    base.gate_all()
    units = []
    repeats: dict[tuple[int, str, int], list[tuple]] = {}
    selected_by_method: dict[tuple[int, str, str], list[int]] = {}
    methods = (METHOD, *COMPARATORS)
    for seed in fresh.SEEDS:
        for target in fresh.TARGETS:
            task, eligible = base.source_task(seed, target)
            allowed = set(eligible.tolist())
            for method in methods:
                root = ROOT if method == METHOD else fresh.ROOT
                rows = base._read(root / str(seed) / target / f"{method}.csv")
                _validate(rows, seed, target, method, allowed)
                if method == METHOD:
                    verify_label_decisions(task, eligible, rows)
                selected_by_method[(seed, target, method)] = [
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
    summary = {method: {target: {metric: float(np.mean([
        unit[metric] for unit in units
        if unit["method"] == method and unit["target"] == target]))
        for metric in METRICS} for target in fresh.TARGETS}
        for method in methods}
    for method in methods:
        summary[method]["overall"] = {metric: float(np.mean([
            unit[metric] for unit in units if unit["method"] == method]))
            for metric in METRICS}
    by_key = {(unit["seed"], unit["target"], unit["method"]): unit
              for unit in units}
    paired = {method: {metric: [[
        by_key[(seed, target, METHOD)][metric]
        - by_key[(seed, target, method)][metric]
        for target in fresh.TARGETS] for seed in fresh.SEEDS]
        for metric in METRICS} for method in COMPARATORS}
    sequence_overlap = []
    for seed in fresh.SEEDS:
        for target in fresh.TARGETS:
            label = selected_by_method[(seed, target, METHOD)]
            continuous = selected_by_method[(seed, target, "ModeShift-Risk")]
            common_prefix = next((index for index, pair in
                                  enumerate(zip(label, continuous, strict=True))
                                  if pair[0] != pair[1]), 50)
            sequence_overlap.append({
                "seed": seed, "target": target,
                "identical_query_prefix": common_prefix,
                "shared_selected_cases": len(set(label) & set(continuous)),
            })
    continuous_necessary = bool(
        all(summary["ModeShift-Risk"][target]["collision_cells"]
            > summary[METHOD][target]["collision_cells"]
            for target in fresh.TARGETS)
        and summary["ModeShift-Risk"]["overall"]["ego_collisions"]
        >= summary[METHOD]["overall"]["ego_collisions"])
    output = {"schema": "mode_label_ablation_development_v1",
              "development_only": True, "budget": 50,
              "seeds": fresh.SEEDS, "targets": fresh.TARGETS,
              "new_physical_target_episodes": len(fresh.SEEDS)
                                               * len(fresh.TARGETS) * 50,
              "continuous_target_feedback_necessary_gate":
              continuous_necessary,
              "repeated_target_scenarios_verified": sum(
                  len(values) > 1 for values in repeats.values()),
              "continuous_vs_label_sequence_overlap": sequence_overlap,
              "summary": summary, "paired_label_minus_comparators": paired,
              "unit_rows": units}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"continuous_necessary": continuous_necessary,
                      "summary": summary, "paired": paired}, indent=2),
          flush=True)
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
