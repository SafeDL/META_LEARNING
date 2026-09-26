"""Seed-clustered, offline comparison of the experimental KDE memory selector.

All outcomes are read from already frozen archives. Hyperparameters are chosen
on two legacy simulator-seed clusters and evaluated on the third; the split is
rotated. No simulator episodes are executed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from method_chains.failure_memory_regression.repair_replay import (
    DEFAULT_CORE, DEFAULT_MEMORY, LEGACY_TARGETS, _family, _load_legacy,
    _regression_seed,
)
from method_chains.failure_memory_regression.selector_v2 import (
    TargetOracle, run_selector_v2,
)
from method_chains.failure_memory_regression.selector_v3 import (
    KernelMemoryConfig, run_memory_kde,
)


OUTPUT = Path("results/method_chains/failure_memory_regression/memory_kde_v3")
REPEATS = 10
TRAIN_REPEATS = 4
BUDGET = 20


def _sign_flip_p(values: list[int]) -> float:
    nonzero = [value for value in values if value]
    if not nonzero:
        return 1.0
    import itertools as it
    observed = abs(sum(nonzero))
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, nonzero)))
                  >= observed
                  for signs in it.product((-1, 1), repeat=len(nonzero)))
    return extreme / (2 ** len(nonzero))


def _task_data(reference: list[dict], candidates_all: list[dict],
               target_banks: dict[str, dict[str, dict]]) -> dict[int, list[dict]]:
    seeds = sorted({int(item["scenario_id"].split(":", 1)[0])
                    for item in candidates_all})
    tasks: dict[int, list[dict]] = {}
    for seed in seeds:
        candidates = [item for item in candidates_all
                      if int(item["scenario_id"].split(":", 1)[0]) == seed]
        history = [row for row in reference if row.get("build_id") == "idm_ref"
                   and row.get("simulator_seed") == seed
                   and row.get("visibility") == "historical"]
        for target in LEGACY_TARGETS:
            target_bank = target_banks.get(target, {})
            evaluator = {item["scenario_id"]: target_bank[item["scenario_id"]]
                         for item in candidates if item["scenario_id"] in target_bank}
            if len(evaluator) != len(candidates):
                raise ValueError(f"incomplete frozen bank: {seed}/{target}")
            failures = sum(row.get("ego_collision") is True and
                           not row.get("inconclusive", False) and
                           next(item for item in candidates
                                if item["scenario_id"] == sid).get("parent_pass") is True
                           for sid, row in evaluator.items())
            tasks.setdefault(seed, []).append({
                "task_id": f"legacy_regression_seed{seed}_{target}",
                "seed": seed, "target": target, "candidates": candidates,
                "history": history, "evaluator": evaluator,
                "failure_pool": failures,
            })
    return tasks


def _method_hits(task: dict, method: str, repeat: int,
                 config: KernelMemoryConfig | None = None) -> int:
    task_id = task["task_id"]
    history = task["history"]
    seed = _regression_seed(task_id, "FBRT-Memory", repeat, REPEATS)
    oracle = TargetOracle(task["evaluator"])
    if method == "FBRT-Memory-KDE-v3":
        queries, _ = run_memory_kde(
            task["candidates"], history, oracle, BUDGET, seed,
            target_build_id=task["target"], parent_build_id="idm_ref",
            mode="regression", session_id=f"{task_id}:KDE:{repeat}",
            config=config)
    else:
        queries, _, _, _ = run_selector_v2(
            method, task["candidates"], history, oracle, BUDGET, seed,
            target_build_id=task["target"], parent_build_id="idm_ref",
            mode="regression", session_id=f"{task_id}:{method}:{repeat}",
            family_by_build={row["build_id"]: row.get("family", _family(row["build_id"]))
                             for row in history})
    return sum(row.get("regression") is True for row in queries)


def _config_grid() -> list[KernelMemoryConfig]:
    values = itertools.product(
        (0.10, 0.18, 0.30),
        (0.35, 0.70, 1.0),
        (0.5, 1.5),
        (0.0, 0.2),
    )
    return [KernelMemoryConfig(bandwidth=h, source_weight=w,
                               source_support_scale=s, exploration=e)
            for h, w, s, e in values]


def run_benchmark(core: Path = DEFAULT_CORE, memory: Path = DEFAULT_MEMORY,
                  output: Path = OUTPUT) -> dict:
    reference, candidates, target_banks = _load_legacy(core)
    tasks_by_seed = _task_data(reference, candidates, target_banks)
    seeds = sorted(tasks_by_seed)
    configs = _config_grid()
    fold_results = []
    heldout_rows = []

    for heldout_seed in seeds:
        train_seeds = [seed for seed in seeds if seed != heldout_seed]
        train_tasks = [task for seed in train_seeds for task in tasks_by_seed[seed]]
        # Paired baseline outcomes are identical across configurations and folds.
        no_memory_cache = {(task["task_id"], repeat): _method_hits(
            task, "FBRT-NoMemory", repeat)
            for task in train_tasks for repeat in range(TRAIN_REPEATS)}
        config_scores = []
        for config in configs:
            hits = 0
            for task in train_tasks:
                for repeat in range(TRAIN_REPEATS):
                    hits += _method_hits(task, "FBRT-Memory-KDE-v3", repeat, config)
            baseline = sum(no_memory_cache.values())
            config_scores.append({
                "config": config.__dict__, "training_hits": hits,
                "training_no_memory_hits": baseline,
                "training_delta": hits - baseline,
            })
        selected_config_row = max(
            config_scores,
            key=lambda row: (row["training_delta"], row["training_hits"],
                             -row["config"]["exploration"],
                             -row["config"]["bandwidth"]))
        config = KernelMemoryConfig(**selected_config_row["config"])

        test_tasks = tasks_by_seed[heldout_seed]
        by_method = {name: 0 for name in (
            "FBRT-Memory-KDE-v3", "FBRT-Memory", "FBRT-NoMemory")}
        repeats_per_task = {name: [] for name in by_method}
        for task in test_tasks:
            for repeat in range(REPEATS):
                for method in by_method:
                    hits = _method_hits(task, method, repeat,
                                        config if method == "FBRT-Memory-KDE-v3" else None)
                    by_method[method] += hits
                    repeats_per_task[method].append({
                        "task_id": task["task_id"], "repeat": repeat,
                        "hits": hits,
                    })
        heldout_rows.extend({"heldout_seed": heldout_seed, **row,
                             "selected_config": config.__dict__}
                            for row in repeats_per_task["FBRT-Memory-KDE-v3"])
        fold_results.append({
            "heldout_seed": heldout_seed, "training_seeds": train_seeds,
            "selected_config": config.__dict__,
            "selected_training_score": selected_config_row,
            "training_grid": config_scores,
            "heldout_total_hits": by_method,
            "heldout_delta_vs_no_memory": (by_method["FBRT-Memory-KDE-v3"] -
                                           by_method["FBRT-NoMemory"]),
            "heldout_delta_vs_memory_v2": (by_method["FBRT-Memory-KDE-v3"] -
                                           by_method["FBRT-Memory"]),
            "heldout_failure_pool": sum(task["failure_pool"] for task in test_tasks),
        })

    cluster_deltas = {}
    for fold in fold_results:
        cluster_deltas[str(fold["heldout_seed"])] = fold["heldout_delta_vs_no_memory"]
    result = {
        "protocol": "offline_leave_one_simulator_seed_cluster_out_v1",
        "input_files": {
            "reference_archive.csv": str(core / "reference_archive.csv"),
            "candidate_pool.csv": str(core / "candidate_pool.csv"),
            "target_response_bank.csv": str(core / "target_response_bank.csv"),
            "archive_v2.jsonl": str(memory / "archive_v2.jsonl"),
        },
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (core / "reference_archive.csv", core / "candidate_pool.csv",
                         core / "target_response_bank.csv", memory / "archive_v2.jsonl")
        },
        "budget": BUDGET, "repeats_per_heldout_task": REPEATS,
        "training_repeats_per_task": TRAIN_REPEATS,
        "folds": fold_results,
        "heldout_cluster_deltas_vs_no_memory": cluster_deltas,
        "sign_flip_p_two_sided": _sign_flip_p(list(cluster_deltas.values())),
        "heldout_seed_clusters": len(seeds),
        "physical_episodes_added": 0,
        "total_heldout_failure_pool": sum(fold["heldout_failure_pool"]
                                           for fold in fold_results),
        "total_heldout_hits": {
            method: sum(fold["heldout_total_hits"][method] for fold in fold_results)
            for method in ("FBRT-Memory-KDE-v3", "FBRT-Memory", "FBRT-NoMemory")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (output / "heldout_queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in heldout_rows), encoding="utf-8")
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
