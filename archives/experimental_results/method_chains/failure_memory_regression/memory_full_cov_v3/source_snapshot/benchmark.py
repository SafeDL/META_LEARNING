"""Offline leave-one-simulator-seed-out comparison of full-covariance memory."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from method_chains.failure_memory_regression.offline_memory_kde_benchmark import (
    DEFAULT_CORE, DEFAULT_MEMORY, REPEATS, BUDGET, _task_data,
)
from method_chains.failure_memory_regression.repair_replay import (
    _family, _load_legacy, _regression_seed,
)
from method_chains.failure_memory_regression.selector_v2 import (
    TargetOracle, run_selector_v2,
)
from method_chains.failure_memory_regression.selector_full_cov_v3 import (
    run_selector_v2 as run_full_cov_selector,
)


OUTPUT = Path("results/method_chains/failure_memory_regression/memory_full_cov_v3")
METHOD = "FBRT-Memory-FullCov-v3"
INITIAL_HISTORY_WEIGHT = 0.5


def _hits(task: dict, method: str, repeat: int) -> int:
    task_id = task["task_id"]
    history = task["history"]
    seed = _regression_seed(task_id, "FBRT-Memory", repeat, REPEATS)
    oracle = TargetOracle(task["evaluator"])
    if method == METHOD:
        queries, _, _, _ = run_full_cov_selector(
            method, task["candidates"], history, oracle, BUDGET, seed,
            target_build_id=task["target"], parent_build_id="idm_ref",
            mode="regression", session_id=f"{task_id}:fullcov:{repeat}",
            family_by_build={row["build_id"]: row.get("family", _family(row["build_id"]))
                             for row in history},
            initial_history_weight=INITIAL_HISTORY_WEIGHT)
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
    fold_results, heldout_queries = [], []
    for heldout_seed, test_tasks in sorted(tasks_by_seed.items()):
        totals = {METHOD: 0, "FBRT-Memory": 0, "FBRT-NoMemory": 0}
        for task in test_tasks:
            for repeat in range(REPEATS):
                row = {"heldout_seed": heldout_seed, "task_id": task["task_id"],
                       "repeat": repeat}
                for method in totals:
                    hits = _hits(task, method, repeat)
                    totals[method] += hits
                    row[method] = hits
                heldout_queries.append(row)
        fold_results.append({
            "heldout_seed": heldout_seed,
            "heldout_total_hits": totals,
            "delta_vs_no_memory": totals[METHOD] - totals["FBRT-NoMemory"],
            "delta_vs_memory_v2": totals[METHOD] - totals["FBRT-Memory"],
            "target_failure_pool": sum(task["failure_pool"] for task in test_tasks),
        })
    deltas = {str(row["heldout_seed"]): row["delta_vs_no_memory"]
              for row in fold_results}
    files = (core / "reference_archive.csv", core / "candidate_pool.csv",
             core / "target_response_bank.csv", memory / "archive_v2.jsonl")
    result = {
        "protocol": "offline_leave_one_simulator_seed_cluster_out_full_covariance_v1",
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
        "budget": BUDGET, "repeats_per_task": REPEATS,
        "initial_history_weight": INITIAL_HISTORY_WEIGHT,
        "folds": fold_results,
        "cluster_deltas_vs_no_memory": deltas,
        "sign_flip_p_two_sided": _sign_flip_p(list(deltas.values())),
        "physical_episodes_added": 0,
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
