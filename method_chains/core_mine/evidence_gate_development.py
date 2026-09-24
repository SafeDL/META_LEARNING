"""Development test of a predeclared static/ModeShift evidence gate."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from method_chains.core_mine import multimode20_experiment as mixed
from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.cell_aware_development import _cell_count
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.data import response_value
from method_chains.core_mine.evidence_gate import gated_scores
from method_chains.core_mine.heterogeneous20_replication import _csv
from method_chains.core_mine.metrics import record_metrics
from method_chains.core_mine.mode_shift_cross_family_replay import FAMILIES
from method_chains.core_mine.oracle import CacheOracle


ROOT = Path("results/method_chains/core_mine/studies/evidence_gate_development")
METHOD = "EvidenceGate"
BASELINES = ("ModeShift-Risk", "ModeQuantile-Static")


def _cache_unit(task, family: str, target: str) -> dict:
    oracle = CacheOracle(task)
    responses: list[float] = []
    events: list[bool] = []
    experts: list[str] = []
    while len(oracle.revealed) < 50:
        scores, expert, _ = gated_scores(
            task, oracle.revealed, responses, events)
        index = choose(scores, oracle.revealed, task.modes, SUPPORT_BUDGET)
        outcome = oracle.reveal(index)
        responses.append(outcome.y)
        events.append(outcome.event)
        experts.append(expert)
    report = record_metrics(task, oracle.revealed, 50)
    return {"family": family, "seed": task.seed, "target": target,
            "method": METHOD, "new_failures": report["NewHistoricalFailureCount"],
            "ego_collisions": report["CollisionCount"],
            "cvs": report["CVS"],
            "static_queries": experts.count("static"),
            "adaptive_queries": experts.count("adaptive"),
            "queried_indices": report["queried_indices"]}


def run_cache() -> Path:
    records = []
    for family, module in FAMILIES.items():
        for seed in module.VALIDATION_SEEDS:
            gate = json.loads((module.ROOT / str(seed) / "gate.json").read_text(
                encoding="utf-8"))
            if not gate["passed"]:
                raise RuntimeError(f"source gate failed {family}/{seed}")
            for target in module.TARGETS:
                records.append(_cache_unit(module._task(seed, target),
                                           family, target))
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "cross_family_records.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"replayed {len(records)} cached family units", flush=True)
    return path


def run_mixed(seed: int, target: str) -> Path:
    mixed.gate_all()
    if seed not in mixed.SEEDS or target not in mixed.TARGETS:
        raise ValueError((seed, target))
    path = ROOT / str(seed) / target / f"{METHOD}.csv"
    if path.exists():
        print(f"reused {seed}/{target}/{METHOD}", flush=True)
        return path
    task, eligible = mixed.source_task(seed, target)
    selected: list[int] = []
    responses: list[float] = []
    events: list[bool] = []
    rows: list[dict] = []
    for query in range(1, 51):
        started = time.perf_counter()
        scores, expert, log_bf = gated_scores(
            task, selected, responses, events, eligible_indices=eligible)
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        started = time.perf_counter()
        result = mixed.target_episode(target, seed, index, task)
        elapsed_seconds = time.perf_counter() - started
        collision = bool(result["ego_collision"])
        near_miss = bool(result["near_miss"])
        event = collision or near_miss
        response = float(response_value(
            np.asarray([result["min_ttc"]]), np.asarray([event]),
            np.asarray([collision]))[0])
        selected.append(index)
        responses.append(response)
        events.append(event)
        rows.append({"seed": seed, "target": target, "method": METHOD,
                     "query": query, "index": index,
                     "mode": str(task.modes[index]),
                     "ego_collision": collision, "near_miss": near_miss,
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response, "expert": expert,
                     "log_bayes_factor": log_bf,
                     "selection_seconds": selection_seconds,
                     "elapsed_seconds": elapsed_seconds})
    path.parent.mkdir(parents=True, exist_ok=True)
    _csv(path, rows)
    print(f"executed {seed}/{target}/{METHOD}: 50", flush=True)
    return path


def _mixed_job(job: tuple[int, str]) -> str:
    return str(run_mixed(*job))


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze() -> dict:
    family_rows = _read(ROOT / "cross_family_records.csv")
    if len(family_rows) != 18:
        raise RuntimeError("all same-family seed/target units are required")
    family_reference = json.loads((Path(
        "results/method_chains/core_mine/studies/mode_shift_cross_family_replay"
    ) / "analysis50.json").read_text(encoding="utf-8"))
    family_summary = {family: {
        metric: float(np.mean([float(row[metric]) for row in family_rows
                               if row["family"] == family]))
        for metric in ("new_failures", "ego_collisions", "cvs",
                       "static_queries", "adaptive_queries")}
        for family in FAMILIES}
    mixed_units = []
    repeats = 0
    for seed in mixed.SEEDS:
        for target in mixed.TARGETS:
            task, eligible = mixed.source_task(seed, target)
            rows = _read(ROOT / str(seed) / target / f"{METHOD}.csv")
            indices = [int(row["index"]) for row in rows]
            if (len(rows) != 50 or len(set(indices)) != 50
                    or set(indices) - set(eligible.tolist())
                    or [int(row["query"]) for row in rows]
                    != list(range(1, 51))
                    or any(row["method"] != METHOD for row in rows)):
                raise RuntimeError(f"invalid mixed B=50 trace {seed}/{target}")
            for baseline in BASELINES:
                old_rows = mixed._read(mixed.ROOT / str(seed) / target
                                       / f"{baseline}.csv")
                if (baseline == "ModeQuantile-Static"
                        and all(row["expert"] == "static" for row in rows)
                        and indices != [int(row["index"]) for row in old_rows]):
                    raise RuntimeError(
                        "all-static gate did not reproduce the frozen static selector")
                old_by_index = {int(row["index"]): row for row in old_rows}
                for row in rows:
                    old = old_by_index.get(int(row["index"]))
                    if old is None:
                        continue
                    fields = ("ego_collision", "near_miss",
                              "background_collision", "completed",
                              "min_ttc", "min_clearance")
                    if any(row[field] != old[field] for field in fields):
                        raise RuntimeError("repeated physical outcome disagrees")
                    repeats += 1
            unit = mixed._unit(task, seed, target, METHOD, rows)
            unit["collision_cells_3x3"] = _cell_count(task, rows, 3)
            unit["collision_cells_5x5"] = _cell_count(task, rows, 5)
            unit["static_queries"] = sum(row["expert"] == "static"
                                          for row in rows)
            unit["adaptive_queries"] = sum(row["expert"] == "adaptive"
                                            for row in rows)
            mixed_units.append(unit)
    mixed_summary = {target: {metric: float(np.mean([
        row[metric] for row in mixed_units if row["target"] == target]))
        for metric in ("collision_cells", "collision_cells_3x3",
                       "collision_cells_5x5", "ego_collisions",
                       "new_failures", "static_queries", "adaptive_queries")}
        for target in mixed.TARGETS}
    original = json.loads((mixed.ROOT / "analysis50.json").read_text(
        encoding="utf-8"))["summary"]
    family_pass = all(
        family_summary[family]["new_failures"] >=
        family_reference["summary"][family]["ModeQuantile-Static"][
            "new_failures"] for family in FAMILIES)
    mixed_cells_pass = all(
        mixed_summary[target]["collision_cells"] >=
        original["ModeShift-Risk"][target]["collision_cells"]
        for target in mixed.TARGETS)
    mixed_collision_count = float(np.mean([
        row["ego_collisions"] for row in mixed_units]))
    retention_pass = mixed_collision_count >= .9 * original[
        "ModeShift-Risk"]["overall"]["ego_collisions"]
    both_experts = any(int(row["static_queries"]) > 0
                       and int(row["adaptive_queries"]) > 0
                       for row in family_rows) and any(
        row["static_queries"] > 0 and row["adaptive_queries"] > 0
        for row in mixed_units)
    passed = bool(family_pass and mixed_cells_pass and retention_pass
                  and both_experts)
    output = {"schema": "evidence_gate_development_v1", "budget": 50,
              "new_physical_target_episodes": 400,
              "same_family_new_physical_target_episodes": 0,
              "same_family_target_banks_inspected_before_protocol": True,
              "source_banks_reused": True,
              "repeated_comparisons_verified": repeats,
              "family_summary": family_summary,
              "family_static_reference": {
                  family: family_reference["summary"][family][
                      "ModeQuantile-Static"] for family in FAMILIES},
              "mixed_summary": mixed_summary,
              "mixed_mode_shift_reference": original["ModeShift-Risk"],
              "development_gate_components": {
                  "same_family": family_pass,
                  "mixed_collision_cells": mixed_cells_pass,
                  "mixed_collision_retention": retention_pass,
                  "both_experts_used": both_experts},
              "development_gate_passed": passed,
              "family_unit_rows": family_rows,
              "mixed_unit_rows": mixed_units}
    (ROOT / "analysis50.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": passed,
                      "components": output["development_gate_components"],
                      "family_summary": family_summary,
                      "mixed_summary": mixed_summary,
                      "repeated_comparisons_verified": repeats},
                     indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "mixed", "analyze", "all"),
                        default="cache")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in {"cache", "all"}:
        run_cache()
    if args.stage in {"mixed", "all"}:
        mixed.gate_all()
        jobs = [(seed, target) for seed in mixed.SEEDS
                for target in mixed.TARGETS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_mixed_job, jobs))
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
