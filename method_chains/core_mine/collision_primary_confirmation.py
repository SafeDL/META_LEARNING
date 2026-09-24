"""Prospective six-seed, 20 Hz, collision-primary B=50 confirmation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from method_chains.core_mine import fvdm_source_robustness as fvdm
from method_chains.core_mine import heterogeneous20_replication as base
from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.control_frequency_audit import _simulate
from method_chains.core_mine.data import response_value
from method_chains.core_mine.simple_residual_ablation import _read, _unit, corrected_scores
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/collision_primary_confirmation")
SEEDS = (20320901, 20320915, 20321006, 20321020, 20321103, 20321117)
SOURCES = ("idm_mobil", "fvdm_ref")
METHODS = (
    ("HistoryMargin-Residual", "mean", True),
    ("HistoryMargin-ModeShift", None, None),
    ("HistoryMargin-Static", "mean", False),
    ("ModeQuantile-Static", None, False),
    ("TargetOnly-Residual", "target", True),
)


def configure() -> None:
    fvdm.ROOT = ROOT
    fvdm.SEEDS = SEEDS
    fvdm._configure()
    if base.ROOT != ROOT or base.SEEDS != SEEDS or base.SOURCES != SOURCES:
        raise RuntimeError("source/target experiment configuration mismatch")


def run_mode_shift(seed: int) -> Path:
    configure()
    path = ROOT / str(seed) / "HistoryMargin-ModeShift.csv"
    if path.exists():
        print(f"reused ModeShift seed={seed}", flush=True)
        return path
    task, eligible = base._task(seed)
    policy = policy_factory(base.TARGET, ASSETS)
    rows: list[dict] = []
    selected: list[int] = []
    outcomes: list[float] = []
    for query in range(1, 51):
        decision_started = time.perf_counter()
        scores = corrected_scores(task, selected, outcomes,
                                  "HistoryMargin-ModeShift")
        index = choose(scores, selected, task.modes, SUPPORT_BUDGET,
                       allowed_indices=eligible)
        selection_seconds = time.perf_counter() - decision_started
        if index in selected or index not in eligible:
            raise RuntimeError("invalid charged target query")
        episode_started = time.perf_counter()
        result = _simulate(base.TARGET, seed, index,
                           float(task.anchors[index, 0]),
                           float(task.anchors[index, 1]),
                           str(task.modes[index]),
                           float(task.controls[index, 0]),
                           float(task.controls[index, 1]),
                           base.FREQUENCY, policy=policy)
        elapsed_seconds = time.perf_counter() - episode_started
        event = bool(result["ego_collision"] or result["near_miss"])
        collision = bool(result["ego_collision"])
        response = float(response_value(
            np.asarray([result["min_ttc"]]),
            np.asarray([event]), np.asarray([collision]))[0])
        selected.append(index)
        outcomes.append(response)
        rows.append({"seed": seed, "method": "HistoryMargin-ModeShift",
                     "query": query, "index": index,
                     "mode": str(task.modes[index]),
                     "ego_collision": collision,
                     "near_miss": bool(result["near_miss"]),
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response,
                     "selection_seconds": selection_seconds,
                     "elapsed_seconds": elapsed_seconds})
    base._csv(path, rows)
    print(f"executed ModeShift seed={seed}: 50 episodes", flush=True)
    return path


def _collision_cells(task, rows: list[dict]) -> int:
    cells = set()
    for row in rows:
        if row["ego_collision"] != "True":
            continue
        xy = task.features[int(row["index"]), :2]
        cell = tuple(np.clip(np.floor(xy * 4).astype(int), 0, 4).tolist())
        cells.add((row["mode"], *cell))
    return len(cells)


def _paired(values: list[float], rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return {"per_seed": array.tolist(), "mean": float(array.mean()),
            "bootstrap_95": [float(np.quantile(samples, .025)),
                             float(np.quantile(samples, .975))]}


def analyze() -> dict:
    configure()
    base._gate_all()
    units = []
    repeats: dict[tuple[int, int], list[tuple]] = {}
    for seed in SEEDS:
        task, eligible = base._task(seed)
        allowed = set(eligible.tolist())
        for method, _, _ in METHODS:
            rows = _read(ROOT / str(seed) / f"{method}.csv")
            indices = [int(row["index"]) for row in rows]
            if (len(rows) != 50 or len(set(indices)) != 50
                    or set(indices) - allowed
                    or [int(row["query"]) for row in rows] != list(range(1, 51))):
                raise RuntimeError(f"invalid charged trace {seed}/{method}")
            for row in rows:
                if (row["event"] == "True") != (
                        row["ego_collision"] == "True"
                        or row["near_miss"] == "True"):
                    raise RuntimeError("event contract mismatch")
                repeats.setdefault((seed, int(row["index"])), []).append(
                    (row["event"], row["ego_collision"], row["near_miss"],
                     float(row["min_ttc"]), float(row["min_clearance"])))
            unit = _unit(task, seed, method, rows)
            unit["collision_cells"] = _collision_cells(task, rows)
            units.append(unit)
    if any(len(set(values)) != 1 for values in repeats.values()):
        raise RuntimeError("cross-method physical replay mismatch")
    by_key = {(row["seed"], row["method"]): row for row in units}
    metrics = ("ego_collisions", "new_failures", "cvs", "collision_cells",
               "failure_modes", "early_auc", "target_wall_seconds",
               "selection_seconds")
    summary = {method: {metric: float(np.mean([
        by_key[(seed, method)][metric] for seed in SEEDS]))
        for metric in metrics} for method, _, _ in METHODS}
    static_names = ("HistoryMargin-Static", "ModeQuantile-Static")
    strongest_static = {seed: max(static_names,
                                  key=lambda method: by_key[(seed, method)][
                                      "ego_collisions"])
                        for seed in SEEDS}
    comparators = {
        "HistoryMargin-ModeShift": ["HistoryMargin-ModeShift"] * len(SEEDS),
        "strongest_static": [strongest_static[seed] for seed in SEEDS],
        "TargetOnly-Residual": ["TargetOnly-Residual"] * len(SEEDS),
    }
    rng = np.random.default_rng(20321201)
    paired = {name: {metric: _paired([
        by_key[(seed, "HistoryMargin-Residual")][metric]
        - by_key[(seed, comparator)][metric]
        for seed, comparator in zip(SEEDS, names, strict=True)], rng)
        for metric in ("ego_collisions", "new_failures", "cvs")}
        for name, names in comparators.items()}
    gate = bool(
        paired["HistoryMargin-ModeShift"]["ego_collisions"]["mean"] > 0
        and paired["HistoryMargin-ModeShift"]["ego_collisions"][
            "bootstrap_95"][0] > 0
        and paired["strongest_static"]["ego_collisions"]["mean"] > 0
        and paired["TargetOnly-Residual"]["ego_collisions"]["mean"] > 0
        and paired["HistoryMargin-ModeShift"]["new_failures"]["mean"] >= -2)
    output = {"schema": "collision_primary_confirmation_v1", "budget": 50,
              "seeds": SEEDS, "source_names": SOURCES, "target": base.TARGET,
              "target_outcomes_precomputed": False,
              "new_physical_target_episodes": len(SEEDS) * len(METHODS) * 50,
              "cross_method_repeated_scenarios_verified": sum(
                  len(values) > 1 for values in repeats.values()),
              "primary_component_gate_passed": gate,
              "strongest_static_by_seed": strongest_static,
              "summary": summary, "paired": paired, "unit_rows": units}
    (ROOT / "analysis50.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"gate": gate, "summary": summary,
                      "paired": paired}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "targets", "analyze", "all"),
                        default="sources")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    configure()
    if args.stage in {"sources", "all"}:
        for seed in SEEDS:
            fvdm.build_sources(seed, args.workers)
    if args.stage in {"targets", "all"}:
        base._gate_all()
        for seed in SEEDS:
            for method, branch, residual in METHODS:
                if method == "HistoryMargin-ModeShift":
                    run_mode_shift(seed)
                else:
                    base.run_campaign(seed, method, branch, residual)
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
