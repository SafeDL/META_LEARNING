"""Build frozen geometry-controlled casebooks after validation calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pearl_learning.src.benchmark_calibration import (
    apply_calibration_manifest,
    episode_near_miss,
    run_baseline_rollout,
    thresholds_for_task,
)
from pearl_learning.src.casebook import (
    CASEBOOK_SCHEMA, CASE_SPLITS, load_casebook, save_casebook, validate_casebook_disjoint,
)
from pearl_learning.src.casebook_v2 import generate_controlled_cases
from pearl_learning.src.io import read_config, write_json
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook


_REACHABLE_COUNTS = {
    "train_pool": 6,
    "validation_support": 2,
    "validation_query": 2,
    "test_support": 2,
    "test_query": 3,
}


def _wanted(task: Any, cfg: dict[str, Any]) -> bool:
    pilot = cfg["method_flow_pilot"]["task_ids"]
    ids = {
        geometry_id
        for split_ids in pilot.values()
        for geometry_id in split_ids
    }
    return task.geometry_id in ids


def _measure(task: Any, cfg: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    env = LogicalMergeEnv(task, cfg, [case])
    try:
        env.reset(options={"case": case})
        return dict(env.initial_case_measurements())
    finally:
        env.close()


def _screen_query(
    task: Any,
    cfg: dict[str, Any],
    case: dict[str, Any],
    *,
    want_heuristic_success: bool,
) -> tuple[bool, dict[str, Any]]:
    thresholds = thresholds_for_task(cfg, task)
    heuristic_row = run_baseline_rollout(task, case, cfg, "heuristic")
    heuristic_outcome = {
        "near_miss": episode_near_miss(heuristic_row["trace"], thresholds),
        "collision": bool(heuristic_row["collision"]),
        "invalid": bool(heuristic_row["invalid"]),
    }
    heuristic_success = heuristic_outcome["near_miss"] and not heuristic_outcome["invalid"]
    if heuristic_success != want_heuristic_success:
        return False, {"heuristic": heuristic_outcome}
    rows = {
        "heuristic": heuristic_row,
        "zero": run_baseline_rollout(task, case, cfg, "zero"),
        "random": run_baseline_rollout(task, case, cfg, "random"),
    }
    outcomes = {
        name: {
            "near_miss": episode_near_miss(row["trace"], thresholds),
            "collision": bool(row["collision"]),
            "invalid": bool(row["invalid"]),
        }
        for name, row in rows.items()
    }
    passive_clean = all(
        not outcomes[name][key]
        for name in ("zero", "random")
        for key in ("near_miss", "collision", "invalid")
    )
    # Calibration trace mode deliberately continues after a candidate near
    # miss. In the executable metric the first such event terminates, so a
    # later trace collision does not invalidate the earlier collision-free hit.
    accepted = passive_clean and heuristic_success == want_heuristic_success
    return accepted, outcomes


def _build_split(task: Any, cfg: dict[str, Any], split: str, calibration_hash: str) -> list[dict[str, Any]]:
    count = int(cfg["cases"]["per_task"][split])
    reachable = int(_REACHABLE_COUNTS[split])
    effective_thresholds = thresholds_for_task(cfg, task)
    active_query_split = (
        (task.split == "meta_validation" and split == "validation_query")
        or (task.split in {"meta_test_template", "meta_test_logical"} and split == "test_query")
    )
    if not active_query_split:
        return generate_controlled_cases(
            task, split, count, cfg,
            arrival_gap_threshold_s=float(effective_thresholds["arrival_gap_threshold_s"]),
            calibration_hash=calibration_hash,
            measure_case=lambda case: _measure(task, cfg, case),
            reachable_count=reachable,
        )

    result: list[dict[str, Any]] = []
    for index in range(count):
        want_success = index < reachable
        accepted = False
        for attempt in range(120):
            candidates = generate_controlled_cases(
                task, split, 1, cfg,
                arrival_gap_threshold_s=float(effective_thresholds["arrival_gap_threshold_s"]),
                calibration_hash=calibration_hash,
                measure_case=lambda case: _measure(task, cfg, case),
                reachable_count=1 if want_success else 0,
                attempt_offset=index * 1000 + attempt,
            )
            case = candidates[0]
            case["case_id"] = f"{task.task_id}_{split}_{index:03d}"
            ok, outcomes = _screen_query(task, cfg, case, want_heuristic_success=want_success)
            if ok:
                case["baseline_screen"] = outcomes
                result.append(case)
                accepted = True
                print(f"accepted {task.task_id}/{split}/{index} after {attempt + 1} attempts", flush=True)
                break
            if (attempt + 1) % 10 == 0:
                print(
                    f"screening {task.task_id}/{split}/{index}: {attempt + 1} rejected; "
                    f"latest={outcomes}",
                    flush=True,
                )
        if not accepted:
            raise RuntimeError(
                f"could not screen {task.task_id}/{split}/{index} after 120 candidates; "
                "do not relax the benchmark gates"
            )
    return result


def _active_case_splits(task: Any, cfg: dict[str, Any]) -> set[str]:
    if task.split == "meta_train":
        gate_ids = set(map(str, cfg.get("physical_heterogeneity_gate", {}).get("task_ids", [])))
        return (
            {"train_pool", "validation_query"}
            if task.geometry_id in gate_ids else {"train_pool"}
        )
    if task.split == "meta_validation":
        return {"validation_support", "validation_query"}
    if task.split in {"meta_test_template", "meta_test_logical"}:
        return {"test_support", "test_query"}
    raise ValueError(f"unsupported task split: {task.split!r}")


def _reuse_casebook(
    task: Any, cfg: dict[str, Any], output_root: str, calibration_hash: str,
) -> tuple[dict[str, list[dict[str, Any]]], str] | None:
    path = Path(output_root) / "casebooks" / f"{task.task_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    active = _active_case_splits(task, cfg)
    provenance = dict(payload.get("provenance", {}))
    if (
        payload.get("schema") != CASEBOOK_SCHEMA
        or provenance.get("calibration_hash") != calibration_hash
        or set(provenance.get("active_case_splits", [])) != active
    ):
        return None
    book = load_casebook(task, output_root, required_schema=CASEBOOK_SCHEMA)
    counts = cfg["cases"]["per_task"]
    if any(len(book.get(split, [])) != (int(counts[split]) if split in active else 0) for split in CASE_SPLITS):
        return None
    if any(
        str(case.get("calibration_hash")) != calibration_hash
        for split in active for case in book[split]
    ):
        return None
    return book, str(payload["sha256"])


def _casebook_provenance(
    task: Any,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    active: set[str],
) -> dict[str, Any]:
    """Record the exact thresholds used to screen this task's cases."""
    profiles = manifest.get("threshold_profiles", {})
    profile_key = (
        str(task.logical_type)
        if str(task.logical_type) in profiles
        else "ood_componentwise_strictest"
    )
    return {
        "calibration_hash": manifest["calibration_hash"],
        "critical_metric_schema": manifest["critical_metric_schema"],
        "thresholds": thresholds_for_task(cfg, task),
        "threshold_profile_key": profile_key,
        "query_screen_policies": ["zero", "random", "heuristic"],
        "active_case_splits": sorted(active),
        "validation_query_purpose": (
            "physical_task_heterogeneity_gate"
            if task.split == "meta_train" and "validation_query" in active
            else "fewshot_evaluation" if "validation_query" in active else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--critical-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids", nargs="*", help="optional geometry-id subset for a staged audit build")
    parser.add_argument("--reuse-existing", action="store_true", help="reuse only fully validated v2 casebooks with the same calibration hash")
    args = parser.parse_args()
    manifest = json.loads(Path(args.critical_thresholds).read_text(encoding="utf-8"))
    cfg = apply_calibration_manifest(read_config(args.config), manifest)
    taskbook = load_taskbook(args.taskbook)
    requested = None if not args.task_ids else set(map(str, args.task_ids))
    tasks = [
        task for values in taskbook.values() for task in values
        if _wanted(task, cfg) and (requested is None or task.geometry_id in requested)
    ]
    if requested is not None and {task.geometry_id for task in tasks} != requested:
        raise ValueError("--task-ids contains a geometry outside the method-flow pilot")
    books: dict[str, dict[str, list[dict[str, Any]]]] = {}
    hashes: dict[str, str] = {}
    for task in tasks:
        active = _active_case_splits(task, cfg)
        reused = _reuse_casebook(task, cfg, args.output, str(manifest["calibration_hash"])) if args.reuse_existing else None
        if reused is not None:
            book, _ = reused
            books[task.task_id] = book
            hashes[task.task_id] = save_casebook(
                task,
                book,
                args.output,
                schema=CASEBOOK_SCHEMA,
                provenance=_casebook_provenance(task, cfg, manifest, active),
            )
            print(f"reused {task.task_id}")
            continue
        book = {
            split: (
                _build_split(task, cfg, split, str(manifest["calibration_hash"]))
                if split in active else []
            )
            for split in CASE_SPLITS
        }
        books[task.task_id] = book
        hashes[task.task_id] = save_casebook(
            task, book, args.output, schema=CASEBOOK_SCHEMA,
            provenance=_casebook_provenance(task, cfg, manifest, active),
        )
        print(f"built {task.task_id}")
    validate_casebook_disjoint(books)
    write_json(Path(args.output) / "casebooks" / "casebook_v2_manifest.json", {
        "schema": CASEBOOK_SCHEMA,
        "calibration_hash": manifest["calibration_hash"],
        "task_casebook_hashes": hashes,
        "task_ids": [task.task_id for task in tasks],
        "static_case_fields_are_network_inputs": False,
    })


if __name__ == "__main__":
    main()
