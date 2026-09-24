"""Frozen multi-target B=50 regression test with collision-bearing modes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from method_chains.core_mine import sparse_sut_experiment as sparse
from method_chains.core_mine.acquisition import choose, marginal_scores
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.data import CachedTask, _features, response_value
from method_chains.core_mine.heterogeneous20_replication import _csv, _sha
from method_chains.core_mine.metrics import cvs
from method_chains.core_mine.oracle import RevealedOutcome
from method_chains.core_mine.posterior import PosteriorModel
from method_chains.core_mine.simple_residual_ablation import corrected_scores
from method_chains.core_mine.single_lane_opportunity_pilot import (
    _external, _profile, _scenario,
)
from method_chains.core_mine.source_safe_development import CONFIG


ROOT = Path("results/method_chains/core_mine/studies/multimode20")
SEEDS = (20330406, 20330420, 20330504, 20330518)
SOURCES = ("idm_mobil", "fvdm_ref")
TARGETS = ("vi_ttc", "fvdm_delay05_brake3")
FIELDS = ("ego_collision", "background_collision", "near_miss", "min_ttc",
          "min_distance", "completed")
METHODS = (
    "MeanGP-Risk", "MeanGP-Marginal", "CoReGP-Risk", "CoReGP-Marginal",
    "TargetGP-Risk", "TargetGP-Marginal", "SourceStatic-Risk",
    "SourceStatic-Marginal", "ModeShift-Risk", "ModeQuantile-Static",
    "RandomSafe",
)
BRANCH = {"MeanGP": "mean", "CoReGP": "composition",
          "TargetGP": "target", "SourceStatic": "mean"}
BOUNDS = {
    "fast_intrusion": ((6.0, 25.0), (-8.0, 8.0)),
    "cutin_braking": ((6.0, 25.0), (-8.0, 8.0)),
    "lead_braking": ((8.0, 30.0), (-7.0, 8.0)),
    "stop_and_go": ((5.0, 24.0), (-8.0, 8.0)),
    "slow_lead_following": ((5.0, 22.0), (-8.0, 7.0)),
}


def proposal(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous = sparse.POOL_VERSION
    try:
        sparse.configure_proposal("v8_source_safe")
        anchors, modes, controls, regimes = sparse.sparse_scenarios(seed)
    finally:
        sparse.configure_proposal(previous)
    if (len(modes) != 320 or len(np.unique(modes)) != 5
            or any(np.sum(modes == mode) != 64 for mode in BOUNDS)):
        raise RuntimeError("frozen v8 proposal must contain five modes x 64")
    for mode, ((gap_low, gap_high), (speed_low, speed_high)) in BOUNDS.items():
        subset = anchors[modes == mode]
        if (np.any(subset[:, 0] < gap_low) or np.any(subset[:, 0] > gap_high)
                or np.any(subset[:, 1] < speed_low)
                or np.any(subset[:, 1] > speed_high)):
            raise RuntimeError(f"proposal bounds changed for {mode}")
    return anchors, modes, controls, regimes


def _source_job(args: tuple[int, str, np.ndarray, np.ndarray,
                            np.ndarray]) -> tuple[str, dict[str, np.ndarray]]:
    seed, source, anchors, modes, controls = args
    values = {field: [] for field in FIELDS}
    for index in range(len(modes)):
        scenario = _scenario(anchors, modes, controls, index)
        outcome = (_external(source, scenario, seed + index)
                   if source == "idm_mobil" else
                   _profile(source, scenario, seed + index))
        for field in FIELDS:
            values[field].append(outcome[field])
    return source, {field: np.asarray(items, dtype=bool if field in {
        "ego_collision", "background_collision", "near_miss", "completed"
    } else float) for field, items in values.items()}


def build_source(seed: int, workers: int) -> dict:
    anchors, modes, controls, regimes = proposal(seed)
    directory = ROOT / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "source_bank.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as bank:
            if (not np.array_equal(bank["anchors"], anchors)
                    or not np.array_equal(bank["modes"], modes)
                    or not np.array_equal(bank["controls"], controls)
                    or tuple(bank["source_names"].astype(str)) != SOURCES
                    or int(bank["ego_control_hz"]) != 20):
                raise RuntimeError("existing source bank violates frozen protocol")
        print(f"reused multimode sources seed={seed}", flush=True)
    else:
        jobs = [(seed, name, anchors, modes, controls) for name in SOURCES]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            batches = dict(pool.map(_source_job, jobs))
        np.savez_compressed(
            path, anchors=anchors, modes=modes, controls=controls,
            regimes=regimes, source_names=np.asarray(SOURCES),
            ego_control_hz=20, physics_hz=20,
            **{field: np.stack([batches[name][field] for name in SOURCES])
               for field in FIELDS})
        print(f"executed multimode sources seed={seed}: 640", flush=True)
    with np.load(path, allow_pickle=False) as bank:
        eligible = ~((bank["ego_collision"] | bank["near_miss"]).any(axis=0))
        eligible &= bank["completed"].all(axis=0)
    by_mode = {mode: int(np.sum(eligible & (modes == mode))) for mode in BOUNDS}
    output = {"seed": seed, "sources": SOURCES, "source_episodes": 640,
              "candidate_pool": 320, "eligible_candidates": int(eligible.sum()),
              "eligible_by_mode": by_mode,
              "passed": bool(eligible.sum() >= 50
                             and all(value >= 24 for value in by_mode.values())),
              "source_bank_sha256": _sha(path),
              "ego_control_hz": 20, "physics_hz": 20,
              "longitudinal_lane_count": 1, "cutin_lane_count": 2}
    (directory / "qualification.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)
    return output


def gate_all() -> None:
    for seed in SEEDS:
        path = ROOT / str(seed) / "qualification.json"
        if not path.exists():
            raise RuntimeError(f"missing source-only gate seed={seed}")
        gate = json.loads(path.read_text(encoding="utf-8"))
        if (not gate["passed"] or gate["sources"] != list(SOURCES)
                or gate["source_bank_sha256"] != _sha(
                    ROOT / str(seed) / "source_bank.npz")):
            raise RuntimeError(f"source-only gate failed seed={seed}")


def source_task(seed: int, target: str) -> tuple[CachedTask, np.ndarray]:
    with np.load(ROOT / str(seed) / "source_bank.npz", allow_pickle=False) as bank:
        anchors, modes, controls = (bank[key].copy() for key in
                                    ("anchors", "modes", "controls"))
        collisions = bank["ego_collision"].copy()
        events = collisions | bank["near_miss"]
        eligible = np.flatnonzero(~events.any(axis=0)
                                  & bank["completed"].all(axis=0))
        source_y = response_value(bank["min_ttc"], events, collisions)
    features, dimensions = _features(anchors, modes, controls)
    n = len(modes)
    task = CachedTask(
        seed, f"Target-{target}-singlelane20", "source_safe", target,
        anchors, modes, controls, features, dimensions,
        source_y, events, collisions, np.zeros(n), np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool), np.full(n, np.inf), np.zeros(n, dtype=bool))
    return task, eligible


def _quantile(task: CachedTask, eligible: np.ndarray) -> np.ndarray:
    source = task.source_y.mean(axis=0)
    scores = np.full(task.count, -np.inf)
    for mode in np.unique(task.modes[eligible]):
        indices = eligible[task.modes[eligible] == mode]
        ranks = rankdata(source[indices], method="average")
        scores[indices] = (ranks - 1) / max(len(indices) - 1, 1)
    return scores


def target_episode(target: str, seed: int, index: int,
                   task: CachedTask) -> dict:
    scenario = _scenario(task.anchors, task.modes, task.controls, index)
    if target == "vi_ttc":
        return _external(target, scenario, seed + index)
    if target == "fvdm_delay05_brake3":
        return _profile(target, scenario, seed + index)
    raise ValueError(target)


def run_campaign(seed: int, target: str, method: str) -> Path:
    gate_all()
    if target not in TARGETS or method not in METHODS:
        raise ValueError((target, method))
    path = ROOT / str(seed) / target / f"{method}.csv"
    if path.exists():
        print(f"reused {seed}/{target}/{method}", flush=True)
        return path
    task, eligible = source_task(seed, target)
    head = method.split("-")[0]
    model = (PosteriorModel(task, CONFIG, BRANCH[head],
                            residual=head != "SourceStatic")
             if head in BRANCH else None)
    quantile = _quantile(task, eligible) if method == "ModeQuantile-Static" else None
    rng = np.random.default_rng(seed + sum(map(ord, target)))
    selected: list[int] = []
    responses: list[float] = []
    severities: list[float] = []
    rows: list[dict] = []
    for query in range(1, 51):
        decision_started = time.perf_counter()
        if model is not None:
            prediction = model.predict()
            if method.endswith("Marginal"):
                scores = marginal_scores(task.features, task.modes, selected,
                                         severities, prediction["p_event"],
                                         prediction["p_collision"], CONFIG.lambda_,
                                         evaluation_indices=eligible)
            else:
                scores = prediction["p_event"]
        elif method == "ModeShift-Risk":
            scores = corrected_scores(task, selected, responses,
                                      "HistoryMargin-ModeShift")
        elif quantile is not None:
            scores = quantile
        else:
            scores = rng.random(task.count)
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - decision_started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        episode_started = time.perf_counter()
        result = target_episode(target, seed, index, task)
        elapsed_seconds = time.perf_counter() - episode_started
        collision = bool(result["ego_collision"])
        near_miss = bool(result["near_miss"])
        event = collision or near_miss
        response = float(response_value(np.asarray([result["min_ttc"]]),
                                        np.asarray([event]),
                                        np.asarray([collision]))[0])
        selected.append(index)
        responses.append(response)
        severities.append(1.0 if collision else .5 if near_miss else 0.0)
        if model is not None:
            model.observe(index, RevealedOutcome(index, response, event,
                                                 collision,
                                                 float(result["min_ttc"])))
        rows.append({"seed": seed, "target": target, "method": method,
                     "query": query, "index": index,
                     "mode": str(task.modes[index]),
                     "ego_collision": collision,
                     "near_miss": near_miss,
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response,
                     "selection_seconds": selection_seconds,
                     "elapsed_seconds": elapsed_seconds})
    path.parent.mkdir(parents=True, exist_ok=True)
    _csv(path, rows)
    print(f"executed {seed}/{target}/{method}: 50", flush=True)
    return path


def _campaign_job(args: tuple[int, str, str]) -> str:
    return str(run_campaign(*args))


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cell(mode: str, gap: float, relative_speed: float) -> tuple[str, int, int]:
    dimensions = BOUNDS[mode]
    cells = []
    for value, (low, high) in zip((gap, relative_speed), dimensions, strict=True):
        if value < low:
            cells.append(-1)
        elif value > high:
            cells.append(4)
        else:
            cells.append(min(3, int(np.floor(4 * (value - low) / (high - low)))))
    return mode, cells[0], cells[1]


def _unit(task: CachedTask, seed: int, target: str, method: str,
          rows: list[dict]) -> dict:
    indices = np.asarray([int(row["index"]) for row in rows], dtype=int)
    events = np.asarray([row["event"] == "True" for row in rows])
    collisions = np.asarray([row["ego_collision"] == "True" for row in rows])
    target_event = np.zeros(task.count, dtype=bool)
    target_collision = np.zeros(task.count, dtype=bool)
    target_event[indices] = events
    target_collision[indices] = collisions
    evaluated = replace(task, target_event=target_event,
                        target_collision=target_collision)
    collision_cells = {_cell(str(task.modes[int(row["index"])]),
                             float(task.anchors[int(row["index"]), 0]),
                             float(task.anchors[int(row["index"]), 1]))
                       for row in rows if row["ego_collision"] == "True"}
    collision_modes = {row["mode"] for row in rows
                       if row["ego_collision"] == "True"}
    failure_modes = {row["mode"] for row in rows if row["event"] == "True"}
    by_mode = {mode: {"queries": sum(row["mode"] == mode for row in rows),
                      "ego_collisions": sum(row["mode"] == mode
                                            and row["ego_collision"] == "True"
                                            for row in rows),
                      "near_misses": sum(row["mode"] == mode
                                         and row["near_miss"] == "True"
                                         for row in rows)} for mode in BOUNDS}
    return {"seed": seed, "target": target, "method": method,
            "collision_cells": len(collision_cells),
            "collision_modes": len(collision_modes),
            "ego_collisions": int(collisions.sum()),
            "new_failures": int(events.sum()),
            "failure_modes": len(failure_modes),
            "cvs": cvs(evaluated, indices),
            "early_auc": float(np.cumsum(events).sum() / 1275),
            "target_wall_seconds": sum(float(row["elapsed_seconds"])
                                       for row in rows),
            "selection_seconds": sum(float(row["selection_seconds"])
                                     for row in rows),
            "by_mode": by_mode}


def _cluster_pair(matrix: np.ndarray, rng: np.random.Generator) -> dict:
    if matrix.shape != (len(SEEDS), len(TARGETS)):
        raise ValueError("paired matrix must retain both targets per seed")
    samples = rng.integers(0, len(SEEDS), size=(10000, len(SEEDS)))
    means = matrix[samples].mean(axis=(1, 2))
    return {"paired_seed_target": matrix.tolist(),
            "mean": float(matrix.mean()),
            "by_target_mean": {target: float(matrix[:, index].mean())
                               for index, target in enumerate(TARGETS)},
            "seed_cluster_bootstrap_95": [float(np.quantile(means, .025)),
                                          float(np.quantile(means, .975))]}


def analyze() -> dict:
    gate_all()
    units = []
    repeats: dict[tuple[int, str, int], list[tuple]] = {}
    for seed in SEEDS:
        for target in TARGETS:
            task, eligible = source_task(seed, target)
            allowed = set(eligible.tolist())
            for method in METHODS:
                rows = _read(ROOT / str(seed) / target / f"{method}.csv")
                indices = [int(row["index"]) for row in rows]
                if (len(rows) != 50 or len(set(indices)) != 50
                        or set(indices) - allowed
                        or [int(row["query"]) for row in rows]
                        != list(range(1, 51))
                        or any(row["method"] != method
                               or row["target"] != target for row in rows)):
                    raise RuntimeError(f"invalid B=50 trace {seed}/{target}/{method}")
                for row in rows:
                    if (row["event"] == "True") != (
                            row["ego_collision"] == "True"
                            or row["near_miss"] == "True"):
                        raise RuntimeError("event response contract changed")
                    key = (seed, target, int(row["index"]))
                    repeats.setdefault(key, []).append(
                        (row["event"], row["ego_collision"], row["near_miss"],
                         row["completed"], float(row["min_ttc"]),
                         float(row["min_clearance"])))
                units.append(_unit(task, seed, target, method, rows))
    if any(len(set(values)) != 1 for values in repeats.values()):
        raise RuntimeError("cross-method repeated physical cases disagree")
    keyed = {(row["seed"], row["target"], row["method"]): row for row in units}
    metric_names = ("collision_cells", "collision_modes", "ego_collisions",
                    "new_failures", "failure_modes", "cvs", "early_auc",
                    "target_wall_seconds", "selection_seconds")
    summary = {method: {target: {metric: float(np.mean([
        keyed[(seed, target, method)][metric] for seed in SEEDS]))
        for metric in metric_names} for target in TARGETS}
        for method in METHODS}
    for method in METHODS:
        summary[method]["overall"] = {metric: float(np.mean([
            keyed[(seed, target, method)][metric]
            for seed in SEEDS for target in TARGETS]))
            for metric in metric_names}
    rng = np.random.default_rng(20330601)
    comparators = ("MeanGP-Risk", "TargetGP-Marginal",
                   "SourceStatic-Marginal", "ModeShift-Risk",
                   "ModeQuantile-Static")
    paired = {}
    for method in comparators:
        paired[method] = {}
        for metric in ("collision_cells", "collision_modes", "ego_collisions",
                       "new_failures", "cvs"):
            matrix = np.asarray([[
                keyed[(seed, target, "MeanGP-Marginal")][metric]
                - keyed[(seed, target, method)][metric]
                for target in TARGETS] for seed in SEEDS], dtype=float)
            paired[method][metric] = _cluster_pair(matrix, rng)
    composition = {}
    for metric in ("collision_cells", "collision_modes", "ego_collisions",
                   "new_failures", "cvs"):
        matrix = np.asarray([[
            keyed[(seed, target, "CoReGP-Marginal")][metric]
            - keyed[(seed, target, "MeanGP-Marginal")][metric]
            for target in TARGETS] for seed in SEEDS], dtype=float)
        composition[metric] = _cluster_pair(matrix, rng)
    primary = "collision_cells"
    component_pass = bool(
        all(paired[name][primary]["mean"] > 0 for name in comparators)
        and all(paired[name][primary]["seed_cluster_bootstrap_95"][0] > 0
                for name in comparators[:3])
        and all(paired[name][primary]["by_target_mean"][target] > 0
                for name in comparators for target in TARGETS)
        and summary["MeanGP-Marginal"]["overall"]["ego_collisions"]
        >= .9 * max(summary[name]["overall"]["ego_collisions"]
                    for name in comparators))
    composition_pass = bool(
        all(composition[primary]["by_target_mean"][target] > 0
            for target in TARGETS)
        and composition[primary]["seed_cluster_bootstrap_95"][0] > 0)
    output = {"schema": "multimode20_b50_v1", "budget": 50,
              "seeds": SEEDS, "sources": SOURCES, "targets": TARGETS,
              "target_outcomes_precomputed": False,
              "new_physical_source_episodes": len(SEEDS) * len(SOURCES) * 320,
              "new_physical_target_episodes": len(SEEDS) * len(TARGETS)
                                               * len(METHODS) * 50,
              "repeated_target_scenarios_verified": sum(
                  len(values) > 1 for values in repeats.values()),
              "mean_gp_marginal_component_gate_passed": component_pass,
              "core_composition_gate_passed": composition_pass,
              "summary": summary, "paired": paired,
              "composition_vs_mean": composition, "unit_rows": units}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    compact = {method: summary[method]["overall"] for method in METHODS}
    print(json.dumps({"gates": {"mean_gp_marginal": component_pass,
                                "core_composition": composition_pass},
                      "overall": compact,
                      "primary_paired": {name: paired[name][primary]
                                         for name in comparators},
                      "composition_primary": composition[primary]},
                     indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "analyze", "all"),
                        default="sources")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in {"sources", "all"}:
        for seed in SEEDS:
            build_source(seed, args.workers)
    if args.stage in {"targets", "all"}:
        gate_all()
        jobs = [(seed, target, method)
                for seed in SEEDS for target in TARGETS for method in METHODS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_campaign_job, jobs))
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
