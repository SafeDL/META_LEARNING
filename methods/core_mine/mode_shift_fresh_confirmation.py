"""Fresh, frozen B=50 confirmation of simple historical mode calibration."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine import multimode20_experiment as base
from methods.core_mine.cell_aware_development import _cell_count


ROOT = Path("results/method_chains/core_mine/studies/mode_shift_fresh_confirmation")
SEEDS = (20330703, 20330717, 20330731, 20330814)
TARGETS = base.TARGETS
SOURCES = base.SOURCES
METHODS = (
    "ModeShift-Risk", "ModeQuantile-Static", "SourceStatic-Marginal",
    "TargetGP-Marginal", "MeanGP-Marginal", "CoReGP-Marginal",
    "RandomSafe",
)
PRIMARY = "collision_cells"


def _configure() -> None:
    base.ROOT = ROOT
    base.SEEDS = SEEDS
    base.METHODS = METHODS


def build_sources(workers: int) -> None:
    _configure()
    for seed in SEEDS:
        gate = base.build_source(seed, workers)
        if not gate["passed"]:
            raise RuntimeError(f"frozen source-only gate failed seed={seed}")


def _target_job(job: tuple[int, str, str]) -> str:
    _configure()
    return str(base.run_campaign(*job))


def run_targets(workers: int) -> None:
    _configure()
    base.gate_all()
    jobs = [(seed, target, method)
            for seed in SEEDS for target in TARGETS for method in METHODS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_target_job, jobs))


def analyze() -> dict:
    _configure()
    base.gate_all()
    units = []
    repeats: dict[tuple[int, str, int], list[tuple]] = {}
    for seed in SEEDS:
        for target in TARGETS:
            task, eligible = base.source_task(seed, target)
            allowed = set(eligible.tolist())
            for method in METHODS:
                rows = base._read(ROOT / str(seed) / target / f"{method}.csv")
                indices = [int(row["index"]) for row in rows]
                if (len(rows) != 50 or len(set(indices)) != 50
                        or set(indices) - allowed
                        or [int(row["query"]) for row in rows]
                        != list(range(1, 51))
                        or any(row["method"] != method
                               or row["target"] != target for row in rows)):
                    raise RuntimeError(
                        f"invalid charged B=50 trace {seed}/{target}/{method}")
                for row in rows:
                    if (row["event"] == "True") != (
                            row["ego_collision"] == "True"
                            or row["near_miss"] == "True"):
                        raise RuntimeError("event response contract changed")
                    key = (seed, target, int(row["index"]))
                    repeats.setdefault(key, []).append((
                        row["event"], row["ego_collision"], row["near_miss"],
                        row["background_collision"], row["completed"],
                        float(row["min_ttc"]),
                        float(row["min_clearance"])))
                unit = base._unit(task, seed, target, method, rows)
                unit["collision_cells_3x3"] = _cell_count(task, rows, 3)
                unit["collision_cells_5x5"] = _cell_count(task, rows, 5)
                units.append(unit)
    if any(len(set(values)) != 1 for values in repeats.values()):
        raise RuntimeError("cross-method repeated physical cases disagree")
    keyed = {(row["seed"], row["target"], row["method"]): row
             for row in units}
    metrics = ("collision_cells", "collision_cells_3x3",
               "collision_cells_5x5", "collision_modes",
               "ego_collisions", "new_failures", "cvs", "early_auc",
               "target_wall_seconds", "selection_seconds")
    summary = {method: {target: {metric: float(np.mean([
        keyed[(seed, target, method)][metric] for seed in SEEDS]))
        for metric in metrics} for target in TARGETS} for method in METHODS}
    for method in METHODS:
        summary[method]["overall"] = {metric: float(np.mean([
            keyed[(seed, target, method)][metric]
            for seed in SEEDS for target in TARGETS])) for metric in metrics}
    rng = np.random.default_rng(20330828)
    paired = {method: {metric: base._cluster_pair(np.asarray([[
        keyed[(seed, target, "ModeShift-Risk")][metric]
        - keyed[(seed, target, method)][metric]
        for target in TARGETS] for seed in SEEDS], dtype=float), rng)
        for metric in metrics} for method in METHODS
        if method != "ModeShift-Risk"}
    static = ("ModeQuantile-Static", "SourceStatic-Marginal")
    target_only = "TargetGP-Marginal"
    primary = "collision_cells"
    passed = bool(
        all(paired[method][primary]["mean"] > 0 for method in paired)
        and all(paired[method][primary]["by_target_mean"][target] > 0
                for method in (*static, target_only) for target in TARGETS)
        and all(paired[method][primary]["seed_cluster_bootstrap_95"][0] > 0
                for method in (*static, target_only))
        and summary["ModeShift-Risk"]["overall"]["ego_collisions"]
        >= .9 * max(summary[method]["overall"]["ego_collisions"]
                    for method in METHODS if method != "ModeShift-Risk")
        and all(not all(summary["ModeShift-Risk"]["overall"][metric]
                        < summary[method]["overall"][metric]
                        for method in (*static, target_only))
                for metric in ("collision_cells_3x3",
                               "collision_cells_5x5")))
    output = {"schema": "mode_shift_fresh_b50_v1", "budget": 50,
              "seeds": SEEDS, "sources": SOURCES, "targets": TARGETS,
              "methods": METHODS, "target_outcomes_precomputed": False,
              "new_physical_source_episodes": len(SEEDS) * len(SOURCES) * 320,
              "new_physical_target_episodes": len(SEEDS) * len(TARGETS)
                                               * len(METHODS) * 50,
              "repeated_target_scenarios_verified": sum(
                  len(values) > 1 for values in repeats.values()),
              "effectiveness_gate_passed": passed,
              "summary": summary, "paired": paired, "unit_rows": units}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis50.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": passed, "overall": {
        method: summary[method]["overall"] for method in METHODS},
        "primary_paired": {method: paired[method][PRIMARY]
                           for method in paired},
        "repeated_target_scenarios_verified":
        output["repeated_target_scenarios_verified"]}, indent=2),
        flush=True)
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
