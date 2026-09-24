"""Diagnostic replay of fixed v8 B=50 target queries at 10/20 Hz control."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from highway_env_benchmark.envs.cutin_env import CutInScenario
from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv
from method_chains.core_mine.replay_source_safe_discoveries import _cases
from replications.highway_sut_selection.runner import ASSETS
from sut_algorithms.highway_env.registry import policy_factory


ROOT = Path("results/method_chains/core_mine/studies/source_safe")
OUTPUT = ROOT / "frequency_audit"
TARGETS = ("mcts_cv", "vi_ttc")
FREQUENCIES = (10, 20)


def _simulate(sut: str, seed: int, index: int, gap: float, relative_speed: float,
              mode: str, timing: float, intensity: float, frequency: int,
              policy=None) -> dict:
    if frequency not in (5, *FREQUENCIES):
        raise ValueError("frequency must divide the 20 Hz physics rate")
    if policy is None:
        policy = policy_factory(sut, ASSETS)
    scenario = CutInScenario(gap, relative_speed, mode, timing, intensity)
    env = ExternalCutInEnv(scenario, ego_kind=policy.ego_kind)
    env.config["policy_frequency"] = frequency
    try:
        env.reset(seed=seed + index)
        policy.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.act(env)
            _, _, terminated, truncated, _ = env.step(action)
        return asdict(env.external_result())
    finally:
        env.close()


def examples() -> None:
    rows = []
    for case in _cases().values():
        for frequency in (5, *FREQUENCIES):
            result = _simulate(case.target, case.seed, case.index, case.gap,
                               case.relative_speed, case.mode, case.timing,
                               case.intensity, frequency)
            if frequency == 5 and (result["ego_collision"] != case.collision
                                   or result["near_miss"] != case.near_miss):
                raise RuntimeError(f"5 Hz reproduction disagrees with bank: {case}")
            rows.append({"seed": case.seed, "target": case.target, "index": case.index,
                         "mode": case.mode, "frequency_hz": frequency,
                         **{key: result[key] for key in
                            ("ego_collision", "near_miss", "min_ttc", "min_distance")}})
            print(f"example {case.mode} {case.target} {frequency} Hz complete", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "examples.csv", rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selected_units() -> list[dict]:
    with (ROOT / "validate" / "records.csv").open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle)
                   if row["method"] == "HistoryMargin-Residual"
                   and row["heterogeneity"] in TARGETS and int(row["repeat"]) == 0]
    if len(records) != 6:
        raise RuntimeError(f"expected six primary validation units, got {len(records)}")
    units = []
    for record in records:
        seed, target = int(record["seed"]), record["heterogeneity"]
        indices = [int(item) for item in record["queried_indices"].split(";")]
        if len(indices) != len(set(indices)) or len(indices) != 50:
            raise RuntimeError("B=50 trace is not 50 unique scenarios")
        with np.load(ROOT / "banks" / "source_safe" /
                     f"sparse_sut_bank_{seed}.npz", allow_pickle=False) as bank:
            target_index = list(bank["sut_names"].astype(str)).index(target)
            source_indices = np.arange(len(bank["sut_names"])) != target_index
            if (bank["ego_collision"][source_indices][:, indices]
                    | bank["near_miss"][source_indices][:, indices]).any():
                raise RuntimeError("originally selected scenario was not source-safe")
            cases = [{"seed": seed, "target": target, "index": index,
                      "gap": float(bank["anchors"][index, 0]),
                      "relative_speed": float(bank["anchors"][index, 1]),
                      "mode": str(bank["modes"][index]),
                      "timing": float(bank["controls"][index, 0]),
                      "intensity": float(bank["controls"][index, 1]),
                      "event_5hz": bool(bank["ego_collision"][target_index, index]
                                        or bank["near_miss"][target_index, index]),
                      "collision_5hz": bool(bank["ego_collision"][target_index, index])}
                     for index in indices]
            units.append({"seed": seed, "target": target, "cases": cases})
    return units


def _unit_job(unit: dict) -> list[dict]:
    policy = policy_factory(unit["target"], ASSETS)
    rows = []
    for case in unit["cases"]:
        for frequency in FREQUENCIES:
            result = _simulate(unit["target"], case["seed"], case["index"],
                               case["gap"], case["relative_speed"], case["mode"],
                               case["timing"], case["intensity"], frequency,
                               policy=policy)
            rows.append({"seed": case["seed"], "target": case["target"],
                         "index": case["index"], "mode": case["mode"],
                         "frequency_hz": frequency, "event_5hz": case["event_5hz"],
                         "collision_5hz": case["collision_5hz"],
                         "event_new": bool(result["ego_collision"] or result["near_miss"]),
                         "collision_new": bool(result["ego_collision"]),
                         "min_ttc_new": result["min_ttc"],
                         "min_clearance_new": result["min_distance"]})
    return rows


def _summary(rows: list[dict]) -> dict:
    groups = {}
    for target in TARGETS:
        for frequency in FREQUENCIES:
            subset = [row for row in rows if row["target"] == target
                      and row["frequency_hz"] == frequency]
            if len(subset) != 150:
                raise RuntimeError(f"incomplete group {target} {frequency}: {len(subset)}")
            old_events = sum(row["event_5hz"] for row in subset)
            persistent = sum(row["event_5hz"] and row["event_new"] for row in subset)
            groups[f"{target}_{frequency}hz"] = {
                "selected_scenarios": len(subset), "events_5hz": old_events,
                "collisions_5hz": sum(row["collision_5hz"] for row in subset),
                "events_new": sum(row["event_new"] for row in subset),
                "collisions_new": sum(row["collision_new"] for row in subset),
                "persistent_5hz_events": persistent,
                "persistence_fraction": persistent / old_events if old_events else None,
                "new_events_on_5hz_safe_queries": sum(
                    not row["event_5hz"] and row["event_new"] for row in subset),
            }
    return {"status": "diagnostic_not_new_B50_validation", "physics_frequency_hz": 20,
            "original_action_frequency_hz": 5, "source_eligibility_rechecked_at_new_rate": False,
            "groups": groups}


def all_selected(workers: int) -> None:
    units = _selected_units()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(units))) as pool:
        futures = {pool.submit(_unit_job, unit): (unit["seed"], unit["target"])
                   for unit in units}
        for future in as_completed(futures):
            seed, target = futures[future]
            rows.extend(future.result())
            rows.sort(key=lambda row: (row["seed"], row["target"], row["index"],
                                       row["frequency_hz"]))
            _write_csv(OUTPUT / "selected_queries.csv", rows)
            print(f"finished seed={seed} target={target}; {len(rows)}/600 replays", flush=True)
    summary = _summary(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("examples", "all"), default="examples")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "examples":
        examples()
    else:
        all_selected(args.workers)


if __name__ == "__main__":
    main()
