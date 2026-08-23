"""Freeze an executable taskbook by resolving every real MetaDrive map."""
from __future__ import annotations

import argparse
from pathlib import Path

from archives.pearl_learning.src.casebook import build_casebook, save_casebook, validate_casebook_disjoint
from archives.pearl_learning.src.io import content_hash, read_config, write_json
from archives.pearl_learning.src.task_env import LogicalMergeEnv
from archives.pearl_learning.src.taskbook import TASKBOOK_SCHEMA, build_taskbook, replace_geometry_hashes, save_taskbook, validate_taskbook


def _resolve_task(task, cfg):
    book = build_casebook(task, cfg)
    probe = next((rows[0] for rows in book.values() if rows), None)
    if probe is None:
        raise ValueError(f"task {task.task_id} has no case available for geometry resolution")
    env = LogicalMergeEnv(task, cfg, [probe], verify_geometry_hash=False)
    try:
        env.reset(options={"case": probe})
        provenance = env.geometry_provenance()
    finally:
        env.close()
    return replace_geometry_hashes(task, **provenance), book


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="directory containing the frozen taskbook; defaults to <output_root>/taskbooks")
    args = parser.parse_args()
    cfg = read_config(args.config)
    root = Path(args.output or Path(cfg["project"]["output_root"]) / "taskbooks")
    candidate = build_taskbook(cfg)
    resolved: dict[str, list] = {split: [] for split in candidate}
    casebooks = {}
    for split, tasks in candidate.items():
        for task in tasks:
            frozen, book = _resolve_task(task, cfg)
            resolved[split].append(frozen)
            casebooks[frozen.task_id] = book
    validate_taskbook(resolved)
    validate_casebook_disjoint(casebooks)
    digest = save_taskbook(resolved, root)
    case_root = root.parent
    case_hashes = {task_id: save_casebook(next(task for tasks in resolved.values() for task in tasks if task.task_id == task_id), book, str(case_root)) for task_id, book in casebooks.items()}
    write_json(root / "taskbook_provenance.json", {
        "schema": TASKBOOK_SCHEMA, "task_schema": "logical_merge_task",
        "taskbook_hash": digest, "casebook_hashes": case_hashes,
        "geometry_catalog_hash": content_hash(cfg["geometry_catalog"]),
    })
    print(f"wrote frozen taskbook sha256={digest} and replayable casebooks")


if __name__ == "__main__":
    main()
