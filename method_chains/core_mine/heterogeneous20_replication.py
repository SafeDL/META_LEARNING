"""Frozen three-seed, physically sequential heterogeneous B=50 validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from method_chains.core_mine import sparse_sut_experiment as sparse
from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.control_frequency_audit import _simulate
from method_chains.core_mine.data import CachedTask, _features, response_value
from method_chains.core_mine.metrics import cvs
from method_chains.core_mine.oracle import RevealedOutcome
from method_chains.core_mine.posterior import PosteriorModel
from method_chains.core_mine.source_safe_development import CONFIG
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/heterogeneous20_replication")
SEEDS = (20320111, 20320125, 20320208)
SOURCES = ("idm_mobil", "mcts_cv")
TARGET = "vi_ttc"
FREQUENCY = 20
FIELDS = ("ego_collision", "background_collision", "near_miss", "min_ttc",
          "min_distance", "completed")
METHODS = (
    ("HistoryMargin-Residual", "mean", True),
    ("CoRe-Residual", "composition", True),
    ("HistoryMargin-Static", "mean", False),
    ("ModeQuantile-Static", None, False),
    ("TargetOnly-Residual", "target", True),
    ("RandomSafe", None, False),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_job(args: tuple[int, str, np.ndarray, np.ndarray,
                            np.ndarray]) -> tuple[str, dict[str, np.ndarray]]:
    seed, sut, anchors, modes, controls = args
    policy = policy_factory(sut, ASSETS)
    values: dict[str, list] = {field: [] for field in FIELDS}
    for index, (anchor, mode, control) in enumerate(zip(anchors, modes, controls,
                                                        strict=True)):
        result = _simulate(sut, seed, index, float(anchor[0]), float(anchor[1]),
                           str(mode), float(control[0]), float(control[1]),
                           FREQUENCY, policy=policy)
        for field in FIELDS:
            values[field].append(result[field])
    return sut, {field: np.asarray(items) for field, items in values.items()}


def build_sources(seed: int, workers: int) -> dict:
    sparse.configure_proposal("v8_source_safe")
    anchors, modes, controls, regimes = sparse.sparse_scenarios(seed)
    if len(modes) != 320 or len(np.unique(modes)) != 5:
        raise RuntimeError("frozen v8 proposal must contain 5 x 64 cases")
    directory = ROOT / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    bank_path = directory / "source_bank.npz"
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as old:
            if not (np.array_equal(old["anchors"], anchors)
                    and np.array_equal(old["modes"], modes)
                    and np.array_equal(old["controls"], controls)
                    and tuple(old["source_names"].astype(str)) == SOURCES
                    and int(old["decision_frequency_hz"]) == FREQUENCY):
                raise RuntimeError("existing source bank differs from frozen protocol")
        print(f"reused sources seed={seed}", flush=True)
    else:
        jobs = [(seed, sut, anchors, modes, controls) for sut in SOURCES]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            batches = dict(pool.map(_source_job, jobs))
        np.savez_compressed(
            bank_path, anchors=anchors, modes=modes, controls=controls,
            regimes=regimes, source_names=np.asarray(SOURCES),
            decision_frequency_hz=FREQUENCY, physics_frequency_hz=20,
            **{field: np.stack([batches[sut][field] for sut in SOURCES])
               for field in FIELDS},
        )
        print(f"executed sources seed={seed}: 640 episodes", flush=True)
    with np.load(bank_path, allow_pickle=False) as bank:
        events = bank["ego_collision"] | bank["near_miss"]
        eligible = ~events.any(axis=0) & bank["completed"].all(axis=0)
        per_mode = {str(mode): int(np.sum(eligible & (bank["modes"] == mode)))
                    for mode in np.unique(bank["modes"])}
    output = {"seed": seed, "sources": SOURCES, "target": TARGET,
              "source_episodes": 640, "decision_frequency_hz": FREQUENCY,
              "physics_frequency_hz": 20, "eligible_candidates": int(eligible.sum()),
              "eligible_by_mode": per_mode,
              "passed": bool(eligible.sum() >= 50 and all(per_mode.values())),
              "source_bank_sha256": _sha(bank_path)}
    (directory / "qualification.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)
    return output


def _gate_all() -> None:
    for seed in SEEDS:
        path = ROOT / str(seed) / "qualification.json"
        if not path.exists():
            raise RuntimeError(f"missing source gate seed={seed}")
        gate = json.loads(path.read_text(encoding="utf-8"))
        if not gate["passed"] or _sha(ROOT / str(seed) / "source_bank.npz") != gate[
                "source_bank_sha256"]:
            raise RuntimeError(f"source gate failed or provenance changed seed={seed}")


def _task(seed: int) -> tuple[CachedTask, np.ndarray]:
    path = ROOT / str(seed) / "source_bank.npz"
    with np.load(path, allow_pickle=False) as bank:
        anchors, modes, controls = (bank[key].copy() for key in
                                    ("anchors", "modes", "controls"))
        collisions = bank["ego_collision"].copy()
        events = collisions | bank["near_miss"]
        eligible = np.flatnonzero(~events.any(axis=0) & bank["completed"].all(axis=0))
        source_y = response_value(bank["min_ttc"], events, collisions)
    features, dimensions = _features(anchors, modes, controls)
    n = len(modes)
    task = CachedTask(seed, f"Target-{TARGET}-heterogeneous20", "source_safe", TARGET,
                      anchors, modes, controls, features, dimensions, source_y,
                      events, collisions, np.zeros(n), np.zeros(n, dtype=bool),
                      np.zeros(n, dtype=bool), np.full(n, np.inf),
                      np.zeros(n, dtype=bool))
    return task, eligible


def _mode_quantile(task: CachedTask, eligible: np.ndarray) -> np.ndarray:
    source = task.source_y.mean(axis=0)
    scores = np.full(task.count, -np.inf)
    for mode in np.unique(task.modes[eligible]):
        indices = eligible[task.modes[eligible] == mode]
        rank = rankdata(source[indices], method="average")
        scores[indices] = (rank - 1) / max(len(indices) - 1, 1)
    return scores


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_campaign(seed: int, method: str, branch: str | None,
                 residual: bool) -> Path:
    _gate_all()
    path = ROOT / str(seed) / f"{method}.csv"
    if path.exists():
        print(f"reused campaign seed={seed} method={method}", flush=True)
        return path
    task, eligible = _task(seed)
    model = PosteriorModel(task, CONFIG, branch, residual) if branch else None
    quantile = _mode_quantile(task, eligible) if method == "ModeQuantile-Static" else None
    rng = np.random.default_rng(seed + sum(map(ord, task.target_name)))
    policy = policy_factory(TARGET, ASSETS)
    rows: list[dict] = []
    selected: list[int] = []
    for query in range(1, 51):
        selection_started = time.perf_counter()
        if model is not None:
            scores = model.predict()["p_event"]
        elif quantile is not None:
            scores = quantile
        else:
            scores = rng.random(task.count)
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - selection_started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        started = time.perf_counter()
        result = _simulate(TARGET, seed, index, float(task.anchors[index, 0]),
                           float(task.anchors[index, 1]), str(task.modes[index]),
                           float(task.controls[index, 0]),
                           float(task.controls[index, 1]), FREQUENCY, policy=policy)
        episode_seconds = time.perf_counter() - started
        event = bool(result["ego_collision"] or result["near_miss"])
        collision = bool(result["ego_collision"])
        response = float(response_value(np.asarray([result["min_ttc"]]),
                                        np.asarray([event]), np.asarray([collision]))[0])
        selected.append(index)
        if model is not None:
            model.observe(index, RevealedOutcome(index, response, event, collision,
                                                 float(result["min_ttc"])))
        rows.append({"seed": seed, "method": method, "query": query,
                     "index": index, "mode": str(task.modes[index]),
                     "ego_collision": collision, "near_miss": bool(result["near_miss"]),
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response,
                     "selection_seconds": selection_seconds,
                     "elapsed_seconds": episode_seconds})
    _csv(path, rows)
    print(f"executed target seed={seed} method={method}: 50 episodes", flush=True)
    return path


def analyze() -> dict:
    _gate_all()
    units = []
    repeats: dict[tuple[int, int], list[tuple[bool, bool, bool, float, float]]] = {}
    for seed in SEEDS:
        task, eligible = _task(seed)
        allowed = set(eligible.tolist())
        for method, _, _ in METHODS:
            with (ROOT / str(seed) / f"{method}.csv").open(
                    encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            indices = [int(row["index"]) for row in rows]
            if (len(rows) != 50 or len(set(indices)) != 50 or set(indices) - allowed
                    or [int(row["query"]) for row in rows] != list(range(1, 51))):
                raise RuntimeError(f"invalid campaign trace seed={seed} method={method}")
            target_event = np.zeros(task.count, dtype=bool)
            target_collision = np.zeros(task.count, dtype=bool)
            for row in rows:
                index = int(row["index"])
                event = row["event"] == "True"
                collision = row["ego_collision"] == "True"
                near_miss = row["near_miss"] == "True"
                if event != (collision or near_miss):
                    raise RuntimeError("event contract changed")
                repeats.setdefault((seed, index), []).append(
                    (event, collision, near_miss, float(row["min_ttc"]),
                     float(row["min_clearance"])))
                target_event[index] = event
                target_collision[index] = collision
            evaluated = replace(task, target_event=target_event,
                                target_collision=target_collision)
            flags = np.asarray([row["event"] == "True" for row in rows])
            collisions = np.asarray([row["ego_collision"] == "True" for row in rows])
            units.append({"seed": seed, "method": method, "new_failures": int(flags.sum()),
                          "ego_collisions": int(collisions.sum()),
                          "failure_modes": len({row["mode"] for row in rows
                                                if row["event"] == "True"}),
                          "early_auc": float(np.cumsum(flags).sum() / 1275),
                          "cvs": cvs(evaluated, np.asarray(indices)),
                          "target_wall_seconds": sum(float(row["elapsed_seconds"])
                                                     for row in rows),
                          "selection_seconds": sum(float(row["selection_seconds"])
                                                   for row in rows)})
    if any(len(set(outcomes)) != 1 for outcomes in repeats.values()):
        raise RuntimeError("repeated physical target executions disagree")
    by_key = {(row["seed"], row["method"]): row for row in units}
    summary = {method: {metric: float(np.mean([by_key[(seed, method)][metric]
                                           for seed in SEEDS]))
                        for metric in ("new_failures", "ego_collisions", "failure_modes",
                                       "early_auc", "cvs", "target_wall_seconds",
                                       "selection_seconds")}
               for method, _, _ in METHODS}
    strongest_static = {seed: max(("HistoryMargin-Static", "ModeQuantile-Static"),
                                  key=lambda name: by_key[(seed, name)]["new_failures"])
                        for seed in SEEDS}
    differences = {}
    for baseline in ("strongest_static", "TargetOnly-Residual", "CoRe-Residual"):
        comparator = [strongest_static[seed] if baseline == "strongest_static" else baseline
                      for seed in SEEDS]
        values = np.asarray([by_key[(seed, "HistoryMargin-Residual")]["new_failures"]
                             - by_key[(seed, name)]["new_failures"]
                             for seed, name in zip(SEEDS, comparator, strict=True)])
        rng = np.random.default_rng(20320301)
        samples = rng.choice(values, size=(10000, len(SEEDS)), replace=True).mean(axis=1)
        differences[baseline] = {"paired_seed_differences": values.tolist(),
                                 "mean": float(values.mean()),
                                 "bootstrap_95": [float(np.quantile(samples, .025)),
                                                  float(np.quantile(samples, .975))]}
    composition_pairs = {}
    for baseline in ("HistoryMargin-Residual", "strongest_static",
                     "TargetOnly-Residual"):
        comparator = [strongest_static[seed] if baseline == "strongest_static" else baseline
                      for seed in SEEDS]
        composition_pairs[baseline] = {}
        for metric in ("new_failures", "ego_collisions", "failure_modes",
                       "early_auc", "cvs"):
            values = np.asarray([by_key[(seed, "CoRe-Residual")][metric]
                                 - by_key[(seed, name)][metric]
                                 for seed, name in zip(SEEDS, comparator, strict=True)])
            rng = np.random.default_rng(20320302)
            samples = rng.choice(values, size=(10000, len(SEEDS)),
                                 replace=True).mean(axis=1)
            composition_pairs[baseline][metric] = {
                "paired_seed_differences": values.tolist(),
                "mean": float(values.mean()),
                "bootstrap_95": [float(np.quantile(samples, .025)),
                                 float(np.quantile(samples, .975))],
            }
    output = {"budget": 50, "seeds": SEEDS, "source_names": SOURCES,
              "target": TARGET, "target_outcomes_precomputed": False,
              "total_target_physical_episodes": len(SEEDS) * len(METHODS) * 50,
              "repeated_target_scenarios_verified": sum(len(outcomes) > 1
                                                         for outcomes in repeats.values()),
              "strongest_static_by_seed": strongest_static,
              "transfer_plus_feedback_mean_gate": bool(
                  differences["strongest_static"]["mean"] > 0
                  and differences["TargetOnly-Residual"]["mean"] > 0),
              "summary": summary, "paired": differences,
              "composition_pairs": composition_pairs, "unit_rows": units}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "analyze", "all"),
                        default="sources")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in ("sources", "all"):
        for seed in SEEDS:
            build_sources(seed, args.workers)
    if args.stage in ("targets", "all"):
        _gate_all()
        for seed in SEEDS:
            for method, branch, residual in METHODS:
                run_campaign(seed, method, branch, residual)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
