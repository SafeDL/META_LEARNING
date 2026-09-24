"""Charged B=50 developmental tests of simpler historical residuals."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from method_chains.core_mine import heterogeneous20_replication as base
from method_chains.core_mine.acquisition import choose
from method_chains.core_mine.config import SUPPORT_BUDGET
from method_chains.core_mine.control_frequency_audit import _simulate
from method_chains.core_mine.data import response_value
from method_chains.core_mine.metrics import cvs
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/simple_residual_ablation")
CONFIGURATIONS = {
    "idm_mcts": {
        "root": Path("results/method_chains/core_mine/studies/heterogeneous20_replication"),
        "seeds": (20320111, 20320125, 20320208),
        "sources": ("idm_mobil", "mcts_cv"),
    },
    "idm_fvdm": {
        "root": Path("results/method_chains/core_mine/studies/fvdm_source_robustness"),
        "seeds": (20320405, 20320419, 20320503),
        "sources": ("idm_mobil", "fvdm_ref"),
    },
}
METHODS = ("HistoryMargin-ModeShift", "HistoryMargin-KernelShift")
LENGTH = 0.30


def configure(name: str) -> dict:
    setting = CONFIGURATIONS[name]
    base.ROOT = setting["root"]
    base.SEEDS = setting["seeds"]
    base.SOURCES = setting["sources"]
    base._gate_all()
    return setting


def corrected_scores(task, selected: list[int], outcomes: list[float],
                     method: str) -> np.ndarray:
    """Only the historical bank and the method's own revealed outcomes enter."""
    source = task.source_y.mean(axis=0)
    scores = source.copy()
    if not selected:
        return scores
    seen = np.asarray(selected, dtype=int)
    residual = np.asarray(outcomes, dtype=float) - source[seen]
    for mode in np.unique(task.modes):
        candidates = np.flatnonzero(task.modes == mode)
        mode_seen = task.modes[seen] == mode
        if not mode_seen.any():
            continue
        values = residual[mode_seen]
        if method == "HistoryMargin-ModeShift":
            scores[candidates] += values.mean()
        elif method == "HistoryMargin-KernelShift":
            delta = task.features[candidates, None, :] - task.features[
                seen[mode_seen]][None, :, :]
            weights = np.exp(-0.5 * np.sum(delta * delta, axis=2) / LENGTH**2)
            scores[candidates] += (weights @ values) / (1.0 + weights.sum(axis=1))
        else:
            raise ValueError(method)
    return scores


def run_campaign(config_name: str, seed: int, method: str) -> Path:
    setting = configure(config_name)
    path = ROOT / config_name / str(seed) / f"{method}.csv"
    if path.exists():
        print(f"reused {config_name} seed={seed} method={method}", flush=True)
        return path
    task, eligible = base._task(seed)
    policy = policy_factory(base.TARGET, ASSETS)
    selected: list[int] = []
    outcomes: list[float] = []
    rows: list[dict] = []
    for query in range(1, 51):
        decision_started = time.perf_counter()
        scores = corrected_scores(task, selected, outcomes, method)
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
        rows.append({"seed": seed, "method": method, "query": query,
                     "index": index, "mode": str(task.modes[index]),
                     "ego_collision": collision,
                     "near_miss": bool(result["near_miss"]),
                     "background_collision": bool(result["background_collision"]),
                     "event": event, "completed": bool(result["completed"]),
                     "min_ttc": float(result["min_ttc"]),
                     "min_clearance": float(result["min_distance"]),
                     "response": response,
                     "selection_seconds": selection_seconds,
                     "elapsed_seconds": elapsed_seconds})
    path.parent.mkdir(parents=True, exist_ok=True)
    base._csv(path, rows)
    print(f"executed {config_name} seed={seed} method={method}: 50", flush=True)
    return path


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _unit(task, seed: int, method: str, rows: list[dict]) -> dict:
    indices = [int(row["index"]) for row in rows]
    events = np.asarray([row["event"] == "True" for row in rows])
    collisions = np.asarray([row["ego_collision"] == "True" for row in rows])
    event_vector = np.zeros(task.count, dtype=bool)
    collision_vector = np.zeros(task.count, dtype=bool)
    event_vector[indices] = events
    collision_vector[indices] = collisions
    evaluated = replace(task, target_event=event_vector,
                        target_collision=collision_vector)
    return {"seed": seed, "method": method, "new_failures": int(events.sum()),
            "ego_collisions": int(collisions.sum()),
            "failure_modes": len({row["mode"] for row in rows
                                  if row["event"] == "True"}),
            "early_auc": float(np.cumsum(events).sum() / 1275),
            "cvs": cvs(evaluated, np.asarray(indices)),
            "target_wall_seconds": sum(float(row["elapsed_seconds"])
                                       for row in rows),
            "selection_seconds": sum(float(row["selection_seconds"])
                                     for row in rows)}


def analyze() -> dict:
    output = {"schema": "simple_residual_developmental_ablation",
              "budget": 50, "target": base.TARGET,
              "developmental_post_hoc": True,
              "target_outcomes_precomputed": False,
              "configurations": {}}
    for name in CONFIGURATIONS:
        setting = configure(name)
        units = []
        repeats: dict[tuple[int, int], list[tuple]] = {}
        for seed in setting["seeds"]:
            task, eligible = base._task(seed)
            allowed = set(eligible.tolist())
            for method in METHODS + ("HistoryMargin-Residual",):
                root = (ROOT / name if method in METHODS else setting["root"])
                rows = _read(root / str(seed) / f"{method}.csv")
                indices = [int(row["index"]) for row in rows]
                if (len(rows) != 50 or len(set(indices)) != 50
                        or set(indices) - allowed
                        or [int(row["query"]) for row in rows] != list(range(1, 51))):
                    raise RuntimeError(f"invalid trace {name}/{seed}/{method}")
                for row in rows:
                    if (row["event"] == "True") != (
                            row["ego_collision"] == "True"
                            or row["near_miss"] == "True"):
                        raise RuntimeError("event contract mismatch")
                    repeats.setdefault((seed, int(row["index"])), []).append(
                        (row["event"], row["ego_collision"], row["near_miss"],
                         float(row["min_ttc"]), float(row["min_clearance"])))
                units.append(_unit(task, seed, method, rows))
        if any(len(set(values)) != 1 for values in repeats.values()):
            raise RuntimeError(f"cross-method physical mismatch in {name}")
        summary = {method: {metric: float(np.mean([
            row[metric] for row in units if row["method"] == method]))
            for metric in ("new_failures", "ego_collisions", "failure_modes",
                           "early_auc", "cvs", "target_wall_seconds",
                           "selection_seconds")}
            for method in METHODS + ("HistoryMargin-Residual",)}
        by_key = {(row["seed"], row["method"]): row for row in units}
        paired = {}
        for method in METHODS:
            paired[method] = {metric: [
                by_key[(seed, method)][metric]
                - by_key[(seed, "HistoryMargin-Residual")][metric]
                for seed in setting["seeds"]]
                for metric in ("new_failures", "ego_collisions", "cvs")}
        output["configurations"][name] = {
            "seeds": setting["seeds"], "source_names": setting["sources"],
            "new_physical_target_episodes": len(setting["seeds"]) * len(METHODS) * 50,
            "cross_method_repeated_scenarios_verified": sum(
                len(values) > 1 for values in repeats.values()),
            "summary": summary, "paired_vs_gp": paired, "unit_rows": units}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "analysis50.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: data["summary"] for name, data
                      in output["configurations"].items()}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("targets", "analyze", "all"),
                        default="targets")
    args = parser.parse_args()
    if args.stage in {"targets", "all"}:
        for name, setting in CONFIGURATIONS.items():
            configure(name)
            for seed in setting["seeds"]:
                for method in METHODS:
                    run_campaign(name, seed, method)
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
