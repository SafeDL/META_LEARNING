"""Screen Gate 3 query-case candidates with the two Gate 1B single-task SACs.

The unscreened four validation_query conditions failed the query oracle audit:
the sut_first single-task SAC achieves no strict VCSR on them.  This script
evaluates a small deterministic candidate grid in the screened train family
and keeps conditions where both tasks reach a strict VCSR with their own SAC
and at least one cross-task policy fails.  This is construction data only:
it is never a test/OOD filter and its provenance is recorded in the frozen
few-shot casebook profile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from archives.pearl_learning.scripts.audit_gate3_query_oracle import _load_sac_model, _select_pair
from archives.pearl_learning.src.benchmark_calibration import resolve_calibration
from archives.pearl_learning.src.io import content_hash, read_config, write_json
from archives.pearl_learning.src.mechanism_casebook import (
    ORDER_BOUNDARY_QUERY_CANDIDATES_V1,
    _condition_rows,
    generate_mechanism_cases,
    validate_matched_mechanism_cases,
)
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import load_taskbook, taskbook_payload
from archives.pearl_learning.scripts.run_baselines import _evaluate_sac


def _measure(task, config, case):
    env = LogicalMergeEnv(task, config, [case])
    try:
        env.reset(options={"case": case})
        return dict(env.initial_case_measurements())
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--sac-checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    args = parser.parse_args()
    cfg = resolve_calibration(read_config(args.config), args.critical_thresholds)
    taskbook = load_taskbook(args.taskbook)
    tasks = _select_pair(taskbook, args.task_id)
    sac_dir = Path(args.sac_checkpoint_dir)
    models = {task.task_id: _load_sac_model(sac_dir, task.task_id) for task in tasks}
    candidates = _condition_rows("order_boundary_query_candidates_v1", ORDER_BOUNDARY_QUERY_CANDIDATES_V1)
    task_a, task_b = tasks
    rows = []
    try:
        for candidate in candidates:
            generated = {
                task.task_id: generate_mechanism_cases(
                    task, cfg, count=1, split="validation_query", conditions=[candidate],
                    measure_case=lambda case, selected=task: _measure(selected, cfg, case),
                )[0]
                for task in tasks
            }
            validate_matched_mechanism_cases(
                {task_id: [case] for task_id, case in generated.items()}
            )
            outcomes = {}
            for source in tasks:
                for target in tasks:
                    result = _evaluate_sac(models[source.task_id], target, cfg, [generated[target.task_id]])
                    outcomes[(source.task_id, target.task_id)] = {
                        "valid_critical_strict": bool(result["records"][0]["valid_critical_strict"]),
                        "episode_return": float(result["records"][0]["episode_return"]),
                    }
            rows.append({
                "condition": candidate,
                "own_policy_vcsr": {
                    task_a.task_id: outcomes[(task_a.task_id, task_a.task_id)],
                    task_b.task_id: outcomes[(task_b.task_id, task_b.task_id)],
                },
                "cross_policy_vcsr": {
                    task_a.task_id: outcomes[(task_b.task_id, task_a.task_id)],
                    task_b.task_id: outcomes[(task_a.task_id, task_b.task_id)],
                },
            })
    finally:
        for model in models.values():
            if getattr(model, "env", None) is not None:
                model.env.close()

    def _own(row: dict, task_id: str) -> bool:
        return bool(row["own_policy_vcsr"][task_id]["valid_critical_strict"])

    # The oracle gate is judged on the four-case aggregate, so select the
    # query set greedily instead of filtering single candidates: start with
    # cases feasible for both tasks, then fill the currently missing side,
    # preferring cases whose own-task policy succeeds.
    both = [row for row in rows if _own(row, task_a.task_id) and _own(row, task_b.task_id)]
    adv_only = [row for row in rows if _own(row, task_a.task_id) and not _own(row, task_b.task_id)]
    sut_only = [row for row in rows if not _own(row, task_a.task_id) and _own(row, task_b.task_id)]
    chosen: list[dict] = list(both[:4])
    while len(chosen) < 4:
        achievable_a = sum(_own(row, task_a.task_id) for row in chosen)
        achievable_b = sum(_own(row, task_b.task_id) for row in chosen)
        pool = adv_only if achievable_a <= achievable_b else sut_only
        if not pool:
            pool = adv_only + sut_only
        if not pool:
            break
        chosen.append(pool.pop(0))
    for row in rows:
        row["selected"] = row in chosen
    selected_conditions = [row["condition"] for row in chosen]

    def _aggregate(rows: list[dict]) -> dict:
        advantage = {
            task_id: (
                float(np.mean([int(_own(row, task_id)) for row in rows]))
                - float(np.mean([bool(row["cross_policy_vcsr"][task_id]["valid_critical_strict"]) for row in rows]))
            )
            for task_id in (task_a.task_id, task_b.task_id)
        }
        achievable = {
            task_id: int(sum(_own(row, task_id) for row in rows))
            for task_id in (task_a.task_id, task_b.task_id)
        }
        return {
            "status": (
                "pass"
                if all(value > 0.0 for value in advantage.values())
                and all(count >= 2 for count in achievable.values())
                else "fail"
            ),
            "diagonal_vcsr_advantage": advantage,
            "query_cases_with_strict_vcsr": achievable,
        }

    aggregate = _aggregate(chosen)
    root = Path(args.output)
    write_json(root / "gate3_query_candidate_screening.json", {
        "schema": "gate3_query_candidate_screening_v1",
        "candidate_count": len(candidates),
        "selected_count": len(chosen),
        "rows": rows,
        "selected_conditions": selected_conditions,
        "selected_aggregate_gate": aggregate,
        "provenance": {
            "taskbook_hash": content_hash(taskbook_payload(taskbook)),
            "config_hash": content_hash(cfg),
            "critical_threshold_hash": cfg["critical_metric"]["calibration_hash"],
            "sac_checkpoint_dir": str(sac_dir),
            "screening_instrument": "gate_1b_single_task_sac_policies",
            "selection_rule": (
                "greedy: both-task feasible cases first, then fill the missing side; "
                "gate judged on the four-case aggregate diagonal advantage"
            ),
            "uses_test_or_ood": False,
            "not_a_benchmark_or_holdout_result": True,
        },
    })
    print(f"query candidate screening: selected {len(chosen)}/{len(candidates)} "
          f"(aggregate gate {aggregate['status']}, advantage {aggregate['diagonal_vcsr_advantage']}, "
          f"achievable {aggregate['query_cases_with_strict_vcsr']})")
    for row in chosen:
        print("  kept", row["condition"]["matched_condition_id"])


if __name__ == "__main__":
    main()
