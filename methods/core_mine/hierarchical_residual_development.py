"""Charged developmental B=50 check of mode-plus-local historical residuals."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine import multimode20_experiment as base
from methods.core_mine.acquisition import choose
from methods.core_mine.config import SUPPORT_BUDGET
from methods.core_mine.data import response_value
from methods.core_mine.hierarchical_residual import HierarchicalResidualModel
from methods.core_mine.heterogeneous20_replication import _csv


ROOT = Path("results/method_chains/core_mine/studies/hierarchical_residual_development")
METHOD = "HierarchicalResidual-Risk"
COMPARATORS = ("ModeShift-Risk", "MeanGP-Risk",
               "ModeQuantile-Static", "TargetGP-Risk")


def run_campaign(seed: int, target: str) -> Path:
    base.gate_all()
    if seed not in base.SEEDS or target not in base.TARGETS:
        raise ValueError((seed, target))
    path = ROOT / str(seed) / target / f"{METHOD}.csv"
    if path.exists():
        print(f"reused hierarchical {seed}/{target}", flush=True)
        return path
    task, eligible = base.source_task(seed, target)
    model = HierarchicalResidualModel(task)
    selected = []
    rows = []
    for query in range(1, 51):
        decision_started = time.perf_counter()
        scores = model.predict()["p_event"]
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - decision_started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        episode_started = time.perf_counter()
        result = base.target_episode(target, seed, index, task)
        elapsed_seconds = time.perf_counter() - episode_started
        collision = bool(result["ego_collision"])
        near_miss = bool(result["near_miss"])
        event = collision or near_miss
        response = float(response_value(np.asarray([result["min_ttc"]]),
                                        np.asarray([event]),
                                        np.asarray([collision]))[0])
        selected.append(index)
        model.observe(index, response)
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
    print(f"executed hierarchical {seed}/{target}: 50", flush=True)
    return path


def _job(args: tuple[int, str]) -> str:
    return str(run_campaign(*args))


def analyze() -> dict:
    base.gate_all()
    units = []
    repeats: dict[tuple[int, str, int], list[tuple]] = {}
    for seed in base.SEEDS:
        for target in base.TARGETS:
            task, eligible = base.source_task(seed, target)
            allowed = set(eligible.tolist())
            for method in (METHOD,) + COMPARATORS:
                root = ROOT if method == METHOD else base.ROOT
                rows = base._read(root / str(seed) / target / f"{method}.csv")
                indices = [int(row["index"]) for row in rows]
                if (len(rows) != 50 or len(set(indices)) != 50
                        or set(indices) - allowed
                        or [int(row["query"]) for row in rows]
                        != list(range(1, 51))):
                    raise RuntimeError(f"invalid campaign {seed}/{target}/{method}")
                for row in rows:
                    repeats.setdefault((seed, target, int(row["index"])), []).append(
                        (row["event"], row["ego_collision"], row["near_miss"],
                         row["completed"], float(row["min_ttc"]),
                         float(row["min_clearance"])))
                units.append(base._unit(task, seed, target, method, rows))
    if any(len(set(values)) != 1 for values in repeats.values()):
        raise RuntimeError("physical repeat mismatch")
    keyed = {(row["seed"], row["target"], row["method"]): row for row in units}
    metrics = ("collision_cells", "collision_modes", "ego_collisions",
               "new_failures", "cvs", "early_auc")
    summary = {method: {target: {metric: float(np.mean([
        keyed[(seed, target, method)][metric] for seed in base.SEEDS]))
        for metric in metrics} for target in base.TARGETS}
        for method in (METHOD,) + COMPARATORS}
    rng = np.random.default_rng(20330701)
    paired = {}
    for comparator in COMPARATORS:
        paired[comparator] = {}
        for metric in metrics:
            matrix = np.asarray([[
                keyed[(seed, target, METHOD)][metric]
                - keyed[(seed, target, comparator)][metric]
                for target in base.TARGETS] for seed in base.SEEDS], dtype=float)
            paired[comparator][metric] = base._cluster_pair(matrix, rng)
    primary = paired["ModeShift-Risk"]["collision_cells"]
    collision = paired["ModeShift-Risk"]["ego_collisions"]
    gate = bool(all(value > 0 for value in primary["by_target_mean"].values())
                and np.sum(np.asarray(primary["paired_seed_target"]) > 0) >= 6
                and collision["mean"] >= 0)
    output = {"schema": "hierarchical_residual_development_v1",
              "developmental_post_hoc": True, "budget": 50,
              "seeds": base.SEEDS, "targets": base.TARGETS,
              "source_bank_root": str(base.ROOT),
              "new_physical_target_episodes": len(base.SEEDS)
                                               * len(base.TARGETS) * 50,
              "repeated_target_scenarios_verified": sum(
                  len(values) > 1 for values in repeats.values()),
              "fresh_seed_confirmation_justified": gate,
              "summary": summary, "paired": paired, "unit_rows": units}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"fresh_seed_confirmation_justified": gate,
                      "summary": summary,
                      "paired_primary": {method: paired[method]["collision_cells"]
                                         for method in COMPARATORS}},
                     indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "analyze", "all"),
                        default="targets")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in {"targets", "all"}:
        base.gate_all()
        jobs = [(seed, target) for seed in base.SEEDS for target in base.TARGETS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_job, jobs))
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
