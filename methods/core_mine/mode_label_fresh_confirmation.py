"""Fresh B=50 comparison of mode-label allocation with static and UCB1."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine import multimode20_experiment as base
from methods.core_mine import mode_label_ablation as label_method
from methods.core_mine.acquisition import choose
from methods.core_mine.cell_aware_development import _cell_count
from methods.core_mine.config import SUPPORT_BUDGET
from methods.core_mine.data import response_value
from methods.core_mine.heterogeneous20_replication import _csv


ROOT = Path("results/method_chains/core_mine/studies/mode_label_fresh_confirmation")
SEEDS = (20330911, 20330925, 20331009, 20331023)
METHOD = "ModeLabelShift-Risk"
METHODS = (METHOD, "SourceMeanRaw-Static", "ModeQuantile-Static", "ModeUCB1")
COMPARATORS = tuple(method for method in METHODS if method != METHOD)
PRIMARY = "collision_cells"


def _configure() -> None:
    base.ROOT = ROOT
    base.SEEDS = SEEDS


def build_sources(workers: int) -> None:
    _configure()
    for seed in SEEDS:
        gate = base.build_source(seed, workers)
        if not gate["passed"]:
            raise RuntimeError(f"frozen source-only gate failed seed={seed}")


def _source_score(task) -> np.ndarray:
    return task.source_y.mean(axis=0)


def _ucb_mode(task, selected: list[int], labels: list[float],
              eligible: np.ndarray) -> str:
    counts = {mode: 0 for mode in base.BOUNDS}
    rewards = {mode: 0.0 for mode in base.BOUNDS}
    for index, label in zip(selected, labels, strict=True):
        mode = str(task.modes[index])
        counts[mode] += 1
        rewards[mode] += label
    remaining = eligible[~np.isin(eligible, np.asarray(selected, dtype=int))]
    available_modes = [mode for mode in base.BOUNDS
                       if np.any(task.modes[remaining] == mode)]
    if not available_modes:
        raise RuntimeError("UCB1 has no eligible candidates remaining")
    untried = [mode for mode in available_modes if counts[mode] == 0]
    if untried:
        return untried[0]
    total = len(selected)
    return max(
        available_modes,
        key=lambda mode: (
            rewards[mode] / counts[mode]
            + math.sqrt(2.0 * math.log(total) / counts[mode]),
            -tuple(base.BOUNDS).index(mode),
        ),
    )


def _choose_index(method: str, task, eligible: np.ndarray,
                  selected: list[int], labels: list[float]) -> int:
    source = _source_score(task)
    if method == "SourceMeanRaw-Static":
        return choose(source, selected, task.modes, SUPPORT_BUDGET,
                      allowed_indices=eligible)
    if method == "ModeQuantile-Static":
        return choose(base._quantile(task, eligible), selected, task.modes,
                      SUPPORT_BUDGET, allowed_indices=eligible)
    if method == METHOD:
        scores = label_method.label_shift_scores(task, selected, labels)
        return choose(scores, selected, task.modes, SUPPORT_BUDGET,
                      allowed_indices=eligible)
    if method == "ModeUCB1":
        if len(selected) < SUPPORT_BUDGET:
            return choose(source, selected, task.modes, SUPPORT_BUDGET,
                          allowed_indices=eligible)
        mode = _ucb_mode(task, selected, labels, eligible)
        available = eligible[
            (task.modes[eligible] == mode)
            & ~np.isin(eligible, np.asarray(selected, dtype=int))
        ]
        if not len(available):
            raise RuntimeError(f"UCB1 selected exhausted mode {mode}")
        return int(available[np.lexsort((available, -source[available]))[0]])
    raise ValueError(method)


def run_campaign(seed: int, target: str, method: str) -> str:
    _configure()
    base.gate_all()
    if target not in base.TARGETS or method not in METHODS:
        raise ValueError((target, method))
    path = ROOT / str(seed) / target / f"{method}.csv"
    if path.exists():
        print(f"reused {seed}/{target}/{method}", flush=True)
        return str(path)

    task, eligible = base.source_task(seed, target)
    selected: list[int] = []
    labels: list[float] = []
    rows: list[dict] = []
    for query in range(1, 51):
        decision_started = time.perf_counter()
        index = _choose_index(method, task, eligible, selected, labels)
        selection_seconds = time.perf_counter() - decision_started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")

        started = time.perf_counter()
        result = base.target_episode(target, seed, index, task)
        elapsed = time.perf_counter() - started
        collision = bool(result["ego_collision"])
        near_miss = bool(result["near_miss"])
        event = collision or near_miss
        label = 0.5 * float(event) + 0.5 * float(collision)
        response = float(response_value(
            np.asarray([result["min_ttc"]]), np.asarray([event]),
            np.asarray([collision]))[0])
        selected.append(index)
        labels.append(label)
        rows.append({
            "seed": seed, "target": target, "method": method,
            "query": query, "index": index, "mode": str(task.modes[index]),
            "ego_collision": collision, "near_miss": near_miss,
            "background_collision": bool(result["background_collision"]),
            "event": event, "completed": bool(result["completed"]),
            "min_ttc": float(result["min_ttc"]),
            "min_clearance": float(result["min_distance"]),
            "response": response, "label_response": label,
            "selection_seconds": selection_seconds,
            "elapsed_seconds": elapsed,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    _csv(path, rows)
    print(f"executed {seed}/{target}/{method}: 50", flush=True)
    return str(path)


def _target_job(job: tuple[int, str, str]) -> str:
    return run_campaign(*job)


def _write_manifest() -> None:
    source_gates = {}
    for seed in SEEDS:
        gate_path = ROOT / str(seed) / "qualification.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate["passed"]:
            raise RuntimeError(f"source-only gate failed seed={seed}")
        source_gates[str(seed)] = {
            "eligible_candidates": gate["eligible_candidates"],
            "eligible_by_mode": gate["eligible_by_mode"],
        }
    manifest = {
        "schema": "mode_label_fresh_confirmation_v1",
        "protocol": "docs/core_mine_mode_label_fresh_confirmation_protocol.md",
        "seeds": SEEDS, "targets": base.TARGETS, "methods": METHODS,
        "budget_per_method_seed_target": 50,
        "support_budget": SUPPORT_BUDGET,
        "target_outcomes_precomputed": False,
        "new_physical_source_episodes": len(SEEDS) * len(base.SOURCES) * 320,
        "new_physical_target_episodes": len(SEEDS) * len(base.TARGETS)
                                        * len(METHODS) * 50,
        "ucb1": {
            "reward": "0.5 * event + 0.5 * ego_collision",
            "exploration_bonus": "sqrt(2 * log(observed_queries) / mode_queries)",
            "within_mode_ranking": "mean_historical_response",
            "mode_tie_break": "frozen BOUNDS insertion order",
        },
        "source_gates": source_gates,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_targets(workers: int) -> None:
    _configure()
    base.gate_all()
    _write_manifest()
    jobs = [(seed, target, method) for seed in SEEDS
            for target in base.TARGETS for method in METHODS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_target_job, jobs))


def _validate_rows(rows: list[dict], seed: int, target: str, method: str,
                   eligible: np.ndarray) -> None:
    indices = [int(row["index"]) for row in rows]
    allowed = set(eligible.tolist())
    if (len(rows) != 50 or len(set(indices)) != 50
            or set(indices) - allowed
            or [int(row["query"]) for row in rows] != list(range(1, 51))
            or any(row["method"] != method or row["target"] != target
                   or int(row["seed"]) != seed for row in rows)):
        raise RuntimeError(f"invalid charged B=50 trace {seed}/{target}/{method}")
    if any((row["event"] == "True") !=
           (row["ego_collision"] == "True" or row["near_miss"] == "True")
           for row in rows):
        raise RuntimeError("event response contract changed")


def verify_decisions(method: str, task, eligible: np.ndarray,
                     rows: list[dict]) -> None:
    selected: list[int] = []
    labels: list[float] = []
    for query, row in enumerate(rows, 1):
        expected = _choose_index(method, task, eligible, selected, labels)
        index = int(row["index"])
        if expected != index:
            raise RuntimeError(f"{method} decision did not replay at query {query}")
        event = row["event"] == "True"
        collision = row["ego_collision"] == "True"
        label = 0.5 * float(event) + 0.5 * float(collision)
        if not np.isclose(float(row["label_response"]), label):
            raise RuntimeError("stored severity label disagrees with target outcome")
        selected.append(index)
        labels.append(label)


def _outcome(row: dict) -> tuple:
    return (row["event"], row["ego_collision"], row["near_miss"],
            row["background_collision"], row["completed"],
            float(row["min_ttc"]), float(row["min_clearance"]))


def analyze() -> dict:
    _configure()
    base.gate_all()
    units = []
    repeats: dict[tuple[int, str, int], list[tuple]] = {}
    for seed in SEEDS:
        for target in base.TARGETS:
            task, eligible = base.source_task(seed, target)
            for method in METHODS:
                rows = base._read(ROOT / str(seed) / target / f"{method}.csv")
                _validate_rows(rows, seed, target, method, eligible)
                verify_decisions(method, task, eligible, rows)
                for row in rows:
                    repeats.setdefault((seed, target, int(row["index"])), [])
                    repeats[(seed, target, int(row["index"]))].append(
                        _outcome(row))
                unit = base._unit(task, seed, target, method, rows)
                unit["collision_cells_3x3"] = _cell_count(task, rows, 3)
                unit["collision_cells_5x5"] = _cell_count(task, rows, 5)
                units.append(unit)
    if any(len(set(values)) != 1 for values in repeats.values()):
        raise RuntimeError("repeated target scenarios disagree across methods")

    keyed = {(unit["seed"], unit["target"], unit["method"]): unit
             for unit in units}
    metrics = ("collision_cells", "collision_cells_3x3",
               "collision_cells_5x5", "collision_modes", "ego_collisions",
               "new_failures", "cvs", "early_auc", "target_wall_seconds",
               "selection_seconds")
    summary = {method: {target: {metric: float(np.mean([
        keyed[(seed, target, method)][metric] for seed in SEEDS]))
        for metric in metrics} for target in base.TARGETS} for method in METHODS}
    for method in METHODS:
        summary[method]["overall"] = {metric: float(np.mean([
            keyed[(seed, target, method)][metric]
            for seed in SEEDS for target in base.TARGETS]))
            for metric in metrics}

    rng = np.random.default_rng(20331030)
    paired = {comparator: {metric: base._cluster_pair(np.asarray([[
        keyed[(seed, target, METHOD)][metric]
        - keyed[(seed, target, comparator)][metric]
        for target in base.TARGETS] for seed in SEEDS], dtype=float), rng)
        for metric in ("collision_cells", "collision_cells_3x3",
                       "collision_cells_5x5", "ego_collisions", "new_failures")}
        for comparator in COMPARATORS}
    gate = bool(
        all(paired[comparator][PRIMARY]["mean"] > 0
            for comparator in COMPARATORS)
        and all(paired[comparator][PRIMARY]["by_target_mean"][target] > 0
                for comparator in COMPARATORS for target in base.TARGETS)
        and all(paired[comparator][PRIMARY]["seed_cluster_bootstrap_95"][0] > 0
                for comparator in COMPARATORS)
        and summary[METHOD]["overall"]["ego_collisions"]
        >= 0.9 * max(summary[comparator]["overall"]["ego_collisions"]
                      for comparator in COMPARATORS)
        and any(summary[METHOD]["overall"][metric]
                >= min(summary[comparator]["overall"][metric]
                       for comparator in COMPARATORS)
                for metric in ("collision_cells_3x3", "collision_cells_5x5")))
    output = {
        "schema": "mode_label_fresh_confirmation_v1", "budget": 50,
        "seeds": SEEDS, "sources": base.SOURCES, "targets": base.TARGETS,
        "methods": METHODS, "target_outcomes_precomputed": False,
        "new_physical_source_episodes": len(SEEDS) * len(base.SOURCES) * 320,
        "new_physical_target_episodes": len(SEEDS) * len(base.TARGETS)
                                        * len(METHODS) * 50,
        "repeated_target_scenarios_verified": sum(
            len(values) > 1 for values in repeats.values()),
        "effectiveness_gate_passed": gate,
        "summary": summary, "paired": paired, "unit_rows": units,
    }
    (ROOT / "analysis50.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate,
                      "overall": {method: summary[method]["overall"]
                                  for method in METHODS},
                      "primary_paired": {method: paired[method][PRIMARY]
                                         for method in COMPARATORS},
                      "repeated_target_scenarios_verified":
                      output["repeated_target_scenarios_verified"]},
                     indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "analyze",
                                             "all"), default="sources")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in {"sources", "all"}:
        build_sources(args.workers)
    if args.stage in {"targets", "all"}:
        run_targets(args.workers)
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
