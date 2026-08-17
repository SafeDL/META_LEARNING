"""Build the isolated matched-case input for Gate 1/2 mechanism audits."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pearl_learning.src.casebook import CASE_SPLITS, MECHANISM_CASEBOOK_SCHEMA, save_casebook
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.mechanism_casebook import (
    MECHANISM_CASEBOOK_PURPOSE,
    MATCHED_PHYSICAL_FIELDS,
    generate_mechanism_cases,
    matched_conditions,
    validate_matched_mechanism_cases,
)
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _measure(task: Any, config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True,
                        help="repeat a geometry id or frozen task id for each matched task")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--profile", choices=("absolute_grid", "order_boundary", "order_boundary_screened_v1"),
                        help="mechanism-only absolute condition profile; defaults to config")
    parser.add_argument("--split", default="train_pool")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if bool(cfg.get("mechanism", {}).get("task_specific_risk_normalization", True)):
        raise ValueError("mechanism casebook must explicitly disable task-specific risk normalization")
    taskbook = load_taskbook(args.taskbook)
    all_tasks = [task for split in taskbook.values() for task in split]
    wanted = set(map(str, args.task_id))
    tasks = [task for task in all_tasks if task.task_id in wanted or task.geometry_id in wanted]
    if len(tasks) != len(wanted):
        raise ValueError("each requested mechanism task must identify exactly one frozen task")
    if len(tasks) != 2:
        raise ValueError("the first mechanism audit intentionally requires exactly two tasks")
    profile = str(args.profile or cfg.get("mechanism", {}).get("case_profile", "absolute_grid"))
    conditions = matched_conditions(args.count, profile=profile)
    selection_provenance = dict(cfg.get("mechanism", {}).get("case_selection", {}))
    books: dict[str, dict[str, list[dict[str, Any]]]] = {}
    hashes: dict[str, str] = {}
    for task in tasks:
        cases = generate_mechanism_cases(
            task, cfg, count=args.count, split=args.split, conditions=conditions,
            measure_case=lambda case, selected=task: _measure(selected, cfg, case),
        )
        book = {name: (cases if name == args.split else []) for name in CASE_SPLITS}
        books[task.task_id] = book
        hashes[task.task_id] = save_casebook(
            task, book, args.output, schema=MECHANISM_CASEBOOK_SCHEMA,
            provenance={
                "schema": MECHANISM_CASEBOOK_SCHEMA,
                "purpose": MECHANISM_CASEBOOK_PURPOSE,
                "task_specific_risk_normalization": False,
                "matched_condition_ids": [row["matched_condition_id"] for row in cases],
                "case_selection": selection_provenance,
                "source_config_hash": content_hash(cfg),
            },
        )
    # Matched task comparisons deliberately reuse the same physical case and
    # simulator seed across the two logical tasks.  Generic split-disjoint
    # validation would reject that required design, so use the stronger
    # condition-by-condition physical equality check instead.
    validate_matched_mechanism_cases({task_id: book[args.split] for task_id, book in books.items()})
    payload = {
        "schema": MECHANISM_CASEBOOK_SCHEMA,
        "purpose": MECHANISM_CASEBOOK_PURPOSE,
        "task_specific_risk_normalization": False,
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "task_casebook_hashes": hashes,
        "task_ids": [task.task_id for task in tasks],
        "split": args.split,
        "conditions": conditions,
        "case_profile": profile,
        "case_selection": selection_provenance,
        "matched_physical_fields": list(MATCHED_PHYSICAL_FIELDS),
    }
    write_json(Path(args.output) / "casebooks" / "mechanism_casebook_manifest.json", payload)
    print(f"built {len(tasks)} matched mechanism casebooks in {args.output}")


if __name__ == "__main__":
    main()
