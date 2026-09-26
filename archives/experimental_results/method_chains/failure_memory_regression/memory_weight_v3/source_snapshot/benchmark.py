"""Leave-one-seed-out tuning of the memory/source prior weight, offline only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from method_chains.failure_memory_regression.offline_memory_kde_benchmark import (
    DEFAULT_CORE, DEFAULT_MEMORY, REPEATS, TRAIN_REPEATS, BUDGET, _task_data,
)
from method_chains.failure_memory_regression.repair_replay import (
    _load_legacy, _family, _regression_seed,
)
from method_chains.failure_memory_regression.selector_v2 import (
    TargetOracle, run_selector_v2,
)
from method_chains.failure_memory_regression.selector_weighted_v3 import (
    run_selector_v2 as run_weighted_selector,
)


OUTPUT = Path("results/method_chains/failure_memory_regression/memory_weight_v3")
WEIGHTS = (0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95)


def _hits(task: dict, method: str, repeat: int, weight: float | None = None) -> int:
    task_id = task["task_id"]
    history = task["history"]
    seed = _regression_seed(task_id, "FBRT-Memory", repeat, REPEATS)
    oracle = TargetOracle(task["evaluator"])
    if method == "FBRT-Memory-Weighted-v3":
        queries, _, _, _ = run_weighted_selector(
            method, task["candidates"], history, oracle, BUDGET, seed,
            target_build_id=task["target"], parent_build_id="idm_ref",
            mode="regression", session_id=f"{task_id}:weighted:{repeat}",
            family_by_build={row["build_id"]: row.get("family", _family(row["build_id"]))
                             for row in history},
            initial_history_weight=float(weight))
    else:
        queries, _, _, _ = run_selector_v2(
            method, task["candidates"], history, oracle, BUDGET, seed,
            target_build_id=task["target"], parent_build_id="idm_ref",
            mode="regression", session_id=f"{task_id}:{method}:{repeat}",
            family_by_build={row["build_id"]: row.get("family", _family(row["build_id"]))
                             for row in history})
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
    tasks_by_seed = _task_data(reference, candidates, target_banks)
    seeds = sorted(tasks_by_seed)
    fold_results = []
    heldout_queries = []

    for heldout_seed in seeds:
        train_tasks = [task for seed in seeds if seed != heldout_seed
                       for task in tasks_by_seed[seed]]
        no_memory_training = sum(_hits(task, "FBRT-NoMemory", repeat)
                                 for task in train_tasks
                                 for repeat in range(TRAIN_REPEATS))
        training = []
        for weight in WEIGHTS:
            hits = sum(_hits(task, "FBRT-Memory-Weighted-v3", repeat, weight)
                       for task in train_tasks for repeat in range(TRAIN_REPEATS))
            training.append({"initial_history_weight": weight,
                             "training_hits": hits,
                             "training_no_memory_hits": no_memory_training,
                             "training_delta": hits - no_memory_training})
        chosen = max(training, key=lambda row: (row["training_delta"], row["training_hits"],
                                                -abs(row["initial_history_weight"] - 0.5)))
        weight = float(chosen["initial_history_weight"])

        test_tasks = tasks_by_seed[heldout_seed]
        totals = {method: 0 for method in (
            "FBRT-Memory-Weighted-v3", "FBRT-Memory", "FBRT-NoMemory")}
        for task in test_tasks:
            for repeat in range(REPEATS):
                weighted_hits = _hits(task, "FBRT-Memory-Weighted-v3", repeat, weight)
                memory_hits = _hits(task, "FBRT-Memory", repeat)
                no_memory_hits = _hits(task, "FBRT-NoMemory", repeat)
                totals["FBRT-Memory-Weighted-v3"] += weighted_hits
                totals["FBRT-Memory"] += memory_hits
                totals["FBRT-NoMemory"] += no_memory_hits
                heldout_queries.append({
                    "heldout_seed": heldout_seed, "task_id": task["task_id"],
                    "repeat": repeat, "weight": weight,
                    "weighted_v3_hits": weighted_hits,
                    "memory_v2_hits": memory_hits,
                    "no_memory_hits": no_memory_hits,
                })
        fold_results.append({
            "heldout_seed": heldout_seed,
            "training_seeds": [seed for seed in seeds if seed != heldout_seed],
            "selected_initial_history_weight": weight,
            "selected_training_result": chosen,
            "training_grid": training,
            "heldout_total_hits": totals,
            "heldout_delta_vs_no_memory": (totals["FBRT-Memory-Weighted-v3"] -
                                           totals["FBRT-NoMemory"]),
            "heldout_delta_vs_memory_v2": (totals["FBRT-Memory-Weighted-v3"] -
                                           totals["FBRT-Memory"]),
            "heldout_failure_pool": sum(task["failure_pool"] for task in test_tasks),
        })

    deltas = {str(row["heldout_seed"]): row["heldout_delta_vs_no_memory"]
              for row in fold_results}
    files = (core / "reference_archive.csv", core / "candidate_pool.csv",
             core / "target_response_bank.csv", memory / "archive_v2.jsonl")
    result = {
        "protocol": "offline_leave_one_simulator_seed_cluster_out_weight_tuning_v1",
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
        "budget": BUDGET, "training_repeats_per_task": TRAIN_REPEATS,
        "heldout_repeats_per_task": REPEATS, "weight_grid": WEIGHTS,
        "folds": fold_results,
        "heldout_cluster_deltas_vs_no_memory": deltas,
        "sign_flip_p_two_sided": _sign_flip_p(list(deltas.values())),
        "heldout_seed_clusters": len(seeds), "physical_episodes_added": 0,
        "total_heldout_failure_pool": sum(row["heldout_failure_pool"]
                                           for row in fold_results),
        "total_heldout_hits": {
            method: sum(row["heldout_total_hits"][method] for row in fold_results)
            for method in ("FBRT-Memory-Weighted-v3", "FBRT-Memory", "FBRT-NoMemory")
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
