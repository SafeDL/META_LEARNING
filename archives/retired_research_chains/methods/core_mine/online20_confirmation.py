"""Truly sequential, physical B=50 confirmation for VI/TTC at 20 Hz."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from methods.core_mine import sparse_sut_experiment as sparse
from methods.core_mine.acquisition import choose
from methods.core_mine.config import SUPPORT_BUDGET
from methods.core_mine.control_frequency_audit import _simulate
from methods.core_mine.data import CachedTask, _features, response_value
from methods.core_mine.oracle import RevealedOutcome
from methods.core_mine.posterior import PosteriorModel
from methods.core_mine.source_safe_development import CONFIG
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


SEED = 20291007
SOURCES = ("idm_mobil", "mcts_cv")
TARGET = "vi_ttc"
FREQUENCY = 20
ROOT = Path("results/method_chains/core_mine/studies/source_safe/online20")
METHODS = (("HistoryMargin-Residual", "mean", True),
           ("HistoryMargin-Static", "mean", False),
           ("TargetOnly-Residual", "target", True),
           ("RandomSafe", None, False))


def _source_job(sut: str, anchors: np.ndarray, modes: np.ndarray,
                controls: np.ndarray) -> dict[str, np.ndarray]:
    policy = policy_factory(sut, ASSETS)
    fields = ("ego_collision", "background_collision", "near_miss", "min_ttc",
              "min_distance", "completed")
    values: dict[str, list] = {field: [] for field in fields}
    for index, (anchor, mode, control) in enumerate(zip(anchors, modes, controls, strict=True)):
        result = _simulate(sut, SEED, index, float(anchor[0]), float(anchor[1]),
                           str(mode), float(control[0]), float(control[1]),
                           FREQUENCY, policy=policy)
        for field in fields:
            values[field].append(result[field])
    return {field: np.asarray(items) for field, items in values.items()}


def build_sources(workers: int) -> dict:
    sparse.configure_proposal("v8_source_safe")
    anchors, modes, controls, regimes = sparse.sparse_scenarios(SEED)
    ROOT.mkdir(parents=True, exist_ok=True)
    bank_path = ROOT / "source_bank.npz"
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as old:
            if not (np.array_equal(old["anchors"], anchors)
                    and np.array_equal(old["modes"], modes)
                    and np.array_equal(old["controls"], controls)
                    and tuple(old["source_names"].astype(str)) == SOURCES):
                raise RuntimeError("existing online20 source bank is incompatible")
        print("reused existing 20 Hz source bank", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(SOURCES))) as pool:
            batches = list(pool.map(_source_job, SOURCES,
                                    [anchors] * len(SOURCES), [modes] * len(SOURCES),
                                    [controls] * len(SOURCES)))
        np.savez_compressed(bank_path, anchors=anchors, modes=modes, controls=controls,
                            regimes=regimes, source_names=np.asarray(SOURCES),
                            policy_frequency_hz=FREQUENCY, physics_frequency_hz=20,
                            **{field: np.stack([batch[field] for batch in batches])
                               for field in batches[0]})
        print("executed 640 source episodes at 20 Hz", flush=True)
    with np.load(bank_path, allow_pickle=False) as bank:
        event = bank["ego_collision"] | bank["near_miss"]
        eligible = (~event.any(axis=0)) & bank["completed"].all(axis=0)
        count = int(eligible.sum())
        modes_found = sorted(set(bank["modes"][eligible].astype(str)))
        background_only = int((bank["background_collision"]
                               & ~bank["ego_collision"]).sum())
    output = {"seed": SEED, "sources": SOURCES, "target": TARGET,
              "decision_frequency_hz": FREQUENCY, "physics_frequency_hz": 20,
              "source_episodes": 640, "eligible_candidates": count,
              "eligible_modes": modes_found, "background_only_source_episodes": background_only,
              "passed": count >= 50 and len(modes_found) == 5}
    (ROOT / "qualification.json").write_text(json.dumps(output, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return output


def _task() -> tuple[CachedTask, np.ndarray]:
    with np.load(ROOT / "source_bank.npz", allow_pickle=False) as bank:
        anchors, modes, controls = (bank[key].copy() for key in
                                    ("anchors", "modes", "controls"))
        collisions = bank["ego_collision"].copy()
        events = collisions | bank["near_miss"]
        eligible = (~events.any(axis=0)) & bank["completed"].all(axis=0)
        source_y = response_value(bank["min_ttc"], events, collisions)
    features, dimensions = _features(anchors, modes, controls)
    count = len(modes)
    # No target outcome is precomputed: these placeholders are never used by
    # PosteriorModel or choose; every actual observation is created on demand.
    task = CachedTask(SEED, f"Target-{TARGET}-online20", "source_safe", TARGET,
                      anchors, modes, controls, features, dimensions,
                      source_y, events, collisions,
                      np.zeros(count), np.zeros(count, dtype=bool),
                      np.zeros(count, dtype=bool), np.full(count, np.inf),
                      np.zeros(count, dtype=bool))
    return task, np.flatnonzero(eligible)


def _campaign(task: CachedTask, eligible: np.ndarray, method: str,
              branch: str | None, residual: bool) -> tuple[list[dict], dict]:
    model = PosteriorModel(task, CONFIG, branch, residual) if branch is not None else None
    policy = policy_factory(TARGET, ASSETS)
    rng = np.random.default_rng(task.seed + sum(map(ord, task.target_name)))
    selected: list[int] = []
    rows = []
    for query in range(1, 51):
        scores = model.predict()["p_event"] if model is not None else rng.random(task.count)
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        if index in selected or index not in eligible:
            raise RuntimeError("invalid online target selection")
        started = time.perf_counter()
        result = _simulate(TARGET, SEED, index, float(task.anchors[index, 0]),
                           float(task.anchors[index, 1]), str(task.modes[index]),
                           float(task.controls[index, 0]),
                           float(task.controls[index, 1]), FREQUENCY, policy=policy)
        elapsed = time.perf_counter() - started
        event = bool(result["ego_collision"] or result["near_miss"])
        collision = bool(result["ego_collision"])
        y = float(response_value(np.asarray([result["min_ttc"]]),
                                 np.asarray([event]), np.asarray([collision]))[0])
        observed = RevealedOutcome(index, y, event, collision, float(result["min_ttc"]))
        selected.append(index)
        if model is not None:
            model.observe(index, observed)
        rows.append({"method": method, "query": query, "seed": SEED,
                     "target": TARGET, "index": index, "mode": str(task.modes[index]),
                     "ego_collision": collision, "near_miss": bool(result["near_miss"]),
                     "event": event, "min_ttc": result["min_ttc"],
                     "min_clearance": result["min_distance"],
                     "elapsed_seconds": elapsed})
    events = np.asarray([row["event"] for row in rows], dtype=bool)
    summary = {"method": method, "budget": 50, "new_target_failures": int(events.sum()),
               "ego_collisions": sum(row["ego_collision"] for row in rows),
               "failure_modes": len({row["mode"] for row in rows if row["event"]}),
               "early_novel_auc": float(np.cumsum(events).sum() / (50 * 51 / 2)),
               "target_physical_episodes": 50,
               "target_wall_seconds": sum(row["elapsed_seconds"] for row in rows)}
    print(json.dumps(summary), flush=True)
    return rows, summary


def confirm() -> None:
    gate_path = ROOT / "qualification.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("online20 confirmation requires a passed source-only gate")
    records_path = ROOT / "target_queries.csv"
    if records_path.exists():
        raise RuntimeError("existing physical confirmation records are preserved; not rerunning")
    task, eligible = _task()
    rows, summaries = [], []
    for method, branch, residual in METHODS:
        observations, summary = _campaign(task, eligible, method, branch, residual)
        rows.extend(observations)
        summaries.append(summary)
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"seed": SEED, "target": TARGET, "sources": SOURCES,
               "budget_per_method": 50, "target_physical_episodes": len(rows),
               "target_outcomes_precomputed": False, "results": summaries}
    (ROOT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n",
                                         encoding="utf-8")
    verify()


def verify() -> dict:
    with (ROOT / "target_queries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    task, eligible = _task()
    allowed = set(eligible.tolist())
    by_method = {}
    by_scenario: dict[int, set[tuple[str, str, str, str]]] = {}
    for method, _, _ in METHODS:
        selected = [row for row in rows if row["method"] == method]
        indices = [int(row["index"]) for row in selected]
        if (len(indices) != 50 or len(set(indices)) != 50
                or set(indices) - allowed
                or [int(row["query"]) for row in selected] != list(range(1, 51))):
            raise RuntimeError(f"invalid online query trace for {method}")
        by_method[method] = {"queries": 50, "unique": 50,
                             "source_safe": True,
                             "functional_modes_queried": sorted(set(task.modes[indices].astype(str)))}
        for row in selected:
            by_scenario.setdefault(int(row["index"]), set()).add(
                (row["event"], row["ego_collision"], row["near_miss"], row["min_ttc"]))
    if len(rows) != 200 or any(len(values) != 1 for values in by_scenario.values()):
        raise RuntimeError("incomplete physical ledger or inconsistent repeated scenario")
    output = {"passed": True, "total_target_physical_episodes": len(rows),
              "distinct_target_scenarios": len(by_scenario),
              "repeated_scenarios_deterministic": True,
              "methods": by_method}
    (ROOT / "verification.json").write_text(json.dumps(output, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "confirm", "verify", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage in ("sources", "all"):
        gate = build_sources(args.workers)
        if args.stage == "all" and not gate["passed"]:
            return
    if args.stage in ("confirm", "all"):
        confirm()
    if args.stage == "verify":
        verify()


if __name__ == "__main__":
    main()
