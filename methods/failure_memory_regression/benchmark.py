"""Offline tuning of memory coverage slots versus posterior exploitation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from methods.failure_memory_regression.replay import (
    DEFAULT_CORE, DEFAULT_MEMORY, LEGACY_TARGETS, _family, _load_legacy,
    _regression_seed,
)
from methods.failure_memory_regression.selector import (
    EXPLOIT_METHOD, TargetOracle, run_selector,
)


OUTPUT = Path("results/method_chains/failure_memory_regression/memory_exploit/coverage_benchmark")
METHOD = EXPLOIT_METHOD
COVERAGE_GRID = (0, 1, 2, 4)
REPEATS = 10
TRAIN_REPEATS = 4
BUDGET = 20


def _tasks_by_seed(reference: list[dict], candidates: list[dict],
                   target_banks: dict[str, dict[str, dict]]) -> dict[int, list[dict]]:
    tasks: dict[int, list[dict]] = {}
    seeds = sorted({int(row["scenario_id"].split(":", 1)[0]) for row in candidates})
    for seed in seeds:
        seed_candidates = [row for row in candidates
                           if int(row["scenario_id"].split(":", 1)[0]) == seed]
        parent_pass = {row["scenario_id"]: row.get("parent_pass") is True
                       for row in seed_candidates}
        history = [row for row in reference if row.get("build_id") == "idm_ref"
                   and row.get("simulator_seed") == seed
                   and row.get("visibility") == "historical"]
        for target in LEGACY_TARGETS:
            bank = target_banks.get(target, {})
            evaluator = {row["scenario_id"]: bank[row["scenario_id"]]
                         for row in seed_candidates if row["scenario_id"] in bank}
            if len(evaluator) != len(seed_candidates):
                raise ValueError(f"incomplete frozen bank: {seed}/{target}")
            failure_pool = sum(row.get("ego_collision") is True
                               and not row.get("inconclusive", False)
                               and parent_pass[scenario_id]
                               for scenario_id, row in evaluator.items())
            tasks.setdefault(seed, []).append({
                "task_id": f"legacy_regression_seed{seed}_{target}",
                "seed": seed, "target": target, "candidates": seed_candidates,
                "history": history, "evaluator": evaluator,
                "failure_pool": failure_pool,
            })
    return tasks


def _hits(task: dict, method: str, repeat: int,
          coverage_slots: int = 2) -> int:
    task_id = task["task_id"]
    history = task["history"]
    seed = _regression_seed(task_id, "FBRT-Memory", repeat, REPEATS)
    oracle = TargetOracle(task["evaluator"])
    session_id = (f"{task_id}:exploit:{repeat}" if method == METHOD else
                  f"{task_id}:{method}:{repeat}")
    queries, _, _, _ = run_selector(
        method, task["candidates"], history, oracle, BUDGET, seed,
        target_build_id=task["target"], parent_build_id="idm_ref",
        mode="regression", session_id=session_id,
        family_by_build={row["build_id"]: row.get("family", _family(row["build_id"]))
                         for row in history},
        coverage_slots=coverage_slots if method == METHOD else None)
    return sum(row.get("regression") is True for row in queries)


def _sign_flip_p(values: list[int]) -> float:
    values = [value for value in values if value]
    if not values:
        return 1.0
    observed = abs(sum(values))
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, values))) >= observed
                  for signs in itertools.product((-1, 1), repeat=len(values)))
    return extreme / (2 ** len(values))


def run_benchmark(core: Path = DEFAULT_CORE, memory: Path = DEFAULT_MEMORY,
                  output: Path = OUTPUT) -> dict:
    reference, candidates, target_banks = _load_legacy(core)
    tasks_by_seed = _tasks_by_seed(reference, candidates, target_banks)
    seeds = sorted(tasks_by_seed)
    fold_results, heldout_queries = [], []
    for heldout_seed in seeds:
        train_tasks = [task for seed in seeds if seed != heldout_seed
                       for task in tasks_by_seed[seed]]
        no_memory_training = sum(_hits(task, "FBRT-NoMemory", repeat)
                                 for task in train_tasks
                                 for repeat in range(TRAIN_REPEATS))
        training = []
        for slots in COVERAGE_GRID:
            hits = sum(_hits(task, METHOD, repeat, slots)
                       for task in train_tasks for repeat in range(TRAIN_REPEATS))
            training.append({"coverage_slots": slots, "training_hits": hits,
                             "training_no_memory_hits": no_memory_training,
                             "training_delta": hits - no_memory_training})
        selected = max(training, key=lambda row: (row["training_delta"],
                                                  row["training_hits"],
                                                  row["coverage_slots"]))
        slots = int(selected["coverage_slots"])
        test_tasks = tasks_by_seed[heldout_seed]
        totals = {METHOD: 0, "FBRT-Memory": 0, "FBRT-NoMemory": 0}
        for task in test_tasks:
            for repeat in range(REPEATS):
                row = {"heldout_seed": heldout_seed, "task_id": task["task_id"],
                       "repeat": repeat, "coverage_slots": slots}
                for method in totals:
                    hits = _hits(task, method, repeat, slots)
                    totals[method] += hits
                    row[method] = hits
                heldout_queries.append(row)
        fold_results.append({
            "heldout_seed": heldout_seed,
            "training_seeds": [seed for seed in seeds if seed != heldout_seed],
            "selected_coverage_slots": slots,
            "selected_training_result": selected, "training_grid": training,
            "heldout_total_hits": totals,
            "delta_vs_no_memory": totals[METHOD] - totals["FBRT-NoMemory"],
            "delta_vs_memory_v2": totals[METHOD] - totals["FBRT-Memory"],
            "target_failure_pool": sum(task["failure_pool"] for task in test_tasks),
        })
    deltas = {str(row["heldout_seed"]): row["delta_vs_no_memory"]
              for row in fold_results}
    task_totals: dict[str, dict[str, int]] = {}
    for row in heldout_queries:
        totals = task_totals.setdefault(row["task_id"],
                                        {METHOD: 0, "FBRT-Memory": 0,
                                         "FBRT-NoMemory": 0})
        for method in totals:
            totals[method] += int(row[method])
    task_delta_no_memory = {task: counts[METHOD] - counts["FBRT-NoMemory"]
                            for task, counts in task_totals.items()}
    task_delta_memory = {task: counts[METHOD] - counts["FBRT-Memory"]
                         for task, counts in task_totals.items()}
    files = (core / "reference_archive.csv", core / "candidate_pool.csv",
             core / "target_response_bank.csv", memory / "archive_v2.jsonl")
    result = {
        "protocol": "offline_leave_one_simulator_seed_cluster_out_coverage_tuning_v1",
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("selector.py"))
        },
        "budget": BUDGET, "training_repeats_per_task": TRAIN_REPEATS,
        "heldout_repeats_per_task": REPEATS, "coverage_slots_grid": COVERAGE_GRID,
        "folds": fold_results, "cluster_deltas_vs_no_memory": deltas,
        "sign_flip_p_two_sided": _sign_flip_p(list(deltas.values())),
        "task_deltas_vs_no_memory": task_delta_no_memory,
        "task_sign_flip_p_vs_no_memory": _sign_flip_p(
            list(task_delta_no_memory.values())),
        "task_deltas_vs_memory_v2": task_delta_memory,
        "task_sign_flip_p_vs_memory_v2": _sign_flip_p(
            list(task_delta_memory.values())),
        "heldout_seed_clusters": len(seeds), "physical_episodes_added": 0,
        "total_target_failure_pool": sum(row["target_failure_pool"]
                                          for row in fold_results),
        "total_hits": {
            method: sum(row["heldout_total_hits"][method] for row in fold_results)
            for method in (METHOD, "FBRT-Memory", "FBRT-NoMemory")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (output / "heldout_queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in heldout_queries), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.core_dir, args.memory_dir, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
