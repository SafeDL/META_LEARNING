"""Measured interaction bank, separate from every frozen FBRT v2 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from highway_sim_env.envs.fbrt_unified_env import run_build_episode
from methods.failure_memory_regression.catalogue import (
    INTERACTION_CATALOGUE, compile_interaction_scenarios,
)
from methods.failure_memory_regression.selector import (
    METHODS, TargetOracle, run_selector,
)
from methods.failure_memory_regression.replay_utils import is_parent_pass

ROOT = Path("results/method_chains/failure_memory_regression/interaction")
BUILDS = ("mobil_rear_guard_off_v2", "mobil_ref_v2", "mobil_rear_state_age",
          "mobil_rear_state_age080",
          "ppo_ref_v2", "ppo_obs_age020_v2")
SEED = 4179901


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def measure(root: Path, builds: tuple[str, ...], limit: int | None = None,
            seed: int = SEED, catalogue_path: Path = INTERACTION_CATALOGUE) -> None:
    cases = compile_interaction_scenarios(root, seed, catalogue_path)
    for build in builds:
        if build not in BUILDS:
            raise ValueError(f"unknown build: {build}")
        path = root / "measured" / f"{build}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = read_jsonl(path)
        existing = {row["scenario_id"]: row for row in previous}
        if len(existing) != len(previous):
            raise ValueError(f"duplicate measured scenario in {path}")
        for case in cases[:limit]:
            if case["scenario_id"] in existing:
                if existing[case["scenario_id"]]["scenario"] != case:
                    raise ValueError(f"cached interaction contract changed: {case['scenario_id']}")
                continue
            result, _ = run_build_episode(build, case, seed)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False) + "\n")
            print(build, case["scenario_id"], "collision", result["ego_collision"],
                  "inconclusive", result["inconclusive"], flush=True)


def evaluate(root: Path, target: str, parent: str | None, budget: int,
             seed: int = SEED, catalogue_path: Path = INTERACTION_CATALOGUE) -> list[dict]:
    cases = compile_interaction_scenarios(root, seed, catalogue_path)
    banks = {build: {row["scenario_id"]: row for row in read_jsonl(
        root / "measured" / f"{build}.jsonl")} for build in BUILDS}
    if any(case["scenario_id"] not in banks[target] for case in cases):
        raise ValueError("target bank is incomplete")
    if parent and any(case["scenario_id"] not in banks[parent] for case in cases):
        raise ValueError("parent bank is incomplete")
    for build in (target, parent) if parent else (target,):
        for case in cases:
            measured = banks[build][case["scenario_id"]]
            if measured["scenario"] != case:
                raise ValueError(f"measured contract mismatch: {build}/{case['scenario_id']}")
    # The historical off build is explicit and never silently enters the target set.
    history = [{**row, "visibility": "historical", "context_id": row["scenario"]["context_id"]}
               for row in banks["mobil_rear_guard_off_v2"].values()]
    if target == "mobil_rear_guard_off_v2":
        history = []
    candidates = [{**case, "scenario": case,
                   "parent_pass": is_parent_pass(banks[parent][case["scenario_id"]])
                   if parent else None} for case in cases]
    if parent:
        candidates = [row for row in candidates if row["parent_pass"]]
    mode = "regression" if parent else "cross_agent"
    oracle_bank = banks[target]
    results = []
    for method in METHODS:
        seeds = range(10) if method == "Random" else range(1)
        for repeat in seeds:
            oracle = TargetOracle(oracle_bank)
            queries, _, _, updates = run_selector(
                method, candidates, history, oracle, budget, 7319 + repeat,
                target_build_id=target, parent_build_id=parent, mode=mode,
                session_id=f"interaction-{target}-{method}-{repeat}")
            results.append({"method": method, "repeat": repeat,
                            "hits": sum(query["selection_reward"] for query in queries),
                            "first_hit": next((query["rank"] for query in queries
                                               if query["selection_reward"]), None),
                            "queries": queries, "updates": updates})
    out = root / "evaluation" / f"{parent or 'cross_agent'}_to_{target}_b{budget}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    for row in results:
        print(row["method"], row["repeat"], row["hits"], row["first_hit"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--catalogue", type=Path, default=INTERACTION_CATALOGUE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--builds", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--target", default="mobil_rear_state_age")
    parser.add_argument("--parent", default="mobil_ref_v2")
    parser.add_argument("--budget", type=int, default=10)
    args = parser.parse_args()
    if args.builds:
        measure(args.root, tuple(args.builds), args.limit, args.seed, args.catalogue)
    if args.evaluate:
        evaluate(args.root, args.target, None if args.parent in {"none", ""} else args.parent,
                 args.budget, args.seed, args.catalogue)


if __name__ == "__main__":
    main()
