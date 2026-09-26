"""Offline-only replay of the frozen FBRT measured result banks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from methods.failure_memory_regression.archive import read_jsonl, row_to_record
from methods.failure_memory_regression.pattern_memory import build_pattern_cards
from methods.failure_memory_regression.replay_utils import (
    contextual_history, excluded_history_counts, history_kind, is_parent_pass,
    is_usable_outcome,
)
from methods.failure_memory_regression.schema import Session, stable_hash
from methods.failure_memory_regression.selector import (
    CHECKPOINTS, EXPLOIT_METHOD, METHODS as BASELINE_METHODS, TargetOracle,
    run_selector,
)


# The replay reads frozen responses without importing the physical episode runner.
PROJECT_RESULTS = Path("results/method_chains/failure_memory_regression")
DEFAULT_CORE = PROJECT_RESULTS / "standard_aligned" / "core"
DEFAULT_MEMORY = PROJECT_RESULTS / "memory_v2"
DEFAULT_OUTPUT = PROJECT_RESULTS / "memory_exploit"
METHODS = (*BASELINE_METHODS, EXPLOIT_METHOD)
REPLAY_BUDGET = 20
RANDOM_REPEATS = 10
LEGACY_TARGETS = ("merge_blind06", "merge_brake2", "slow_front_brake2")
COMPACT_BUILD_PAIRS = (
    ("mobil_ref_v2", "mobil_rear_guard_off_v2",
     "compact_regression_mobil_ref_to_rear_guard_off"),
    ("ppo_ref_v2", "ppo_obs_age020_v2",
     "compact_regression_ppo_ref_to_obs_age020"),
)
FINGERPRINT_FILES = (
    "reference_archive.csv", "candidate_pool.csv", "target_response_bank.csv",
)
FINGERPRINT_MEMORY_FILES = (
    "archive_v2.jsonl", "summary_by_task.csv", "report.md", "acceptance.json",
    "compact_bank/scenario_cases.jsonl", "compact_bank/episodes.jsonl",
)
FINGERPRINT_MEMORY_DIRS = ("sessions", "validation_audit_v4")
CODE_FILES = (
    "methods/failure_memory_regression/archive.py",
    "methods/failure_memory_regression/bayes_model.py",
    "methods/failure_memory_regression/experiment.py",
    "methods/failure_memory_regression/pattern_memory.py",
    "methods/failure_memory_regression/replay_utils.py",
    "methods/failure_memory_regression/selector.py",
    "methods/failure_memory_regression/replay.py",
)


def _regression_seed(task_id: str, method: str, repeat: int,
                     paired_repeats: int) -> int:
    """Use common random numbers across methods in an explicit paired audit."""
    label = "paired" if paired_repeats > 1 else method
    return 4179901 + repeat + int(stable_hash(f"{task_id}:{label}:{repeat}")[:8], 16)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return read_jsonl(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8-sig")
    else:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                                 if isinstance(value, (dict, list, tuple)) else value
                                 for key, value in row.items()})
    os.replace(temporary, path)


def _family(build_id: str) -> str:
    if build_id == "idm_ref":
        return "legacy_profiled_idm_reference"
    if build_id in LEGACY_TARGETS:
        return f"legacy_profiled_idm_fault:{build_id}"
    if build_id.startswith("mobil_"):
        return "native_idm_mobil"
    if build_id.startswith("ppo_"):
        return "ppo_ece"
    return build_id


def _record(row: dict, source: str) -> dict:
    result = row_to_record(row, source).as_dict()
    result["family"] = _family(result["build_id"])
    return result


def _snapshot_hash(rows: list[dict]) -> str:
    return stable_hash(sorted(rows, key=lambda row: row.get("execution_id", "")))


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_manifest(core: Path, memory: Path) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    file_paths = [*(core / name for name in FINGERPRINT_FILES),
                  *(memory / name for name in FINGERPRINT_MEMORY_FILES)]
    for path in file_paths:
        label = path.as_posix()
        if path.is_file():
            hashes[label] = _sha256(path)
        else:
            missing.append(label)
    for dirname in FINGERPRINT_MEMORY_DIRS:
        directory = memory / dirname
        if not directory.is_dir():
            missing.append(directory.as_posix())
            continue
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not files:
            missing.append(directory.as_posix() + " (empty)")
        for path in files:
            hashes[path.as_posix()] = _sha256(path)
    return hashes, missing


def _required_missing(core: Path, memory: Path) -> list[str]:
    missing = [path.as_posix() for path in
               [*(core / name for name in FINGERPRINT_FILES),
                memory / "archive_v2.jsonl",
                memory / "compact_bank" / "scenario_cases.jsonl",
                memory / "compact_bank" / "episodes.jsonl"] if not path.is_file()]
    return missing


def check_inputs(core: Path = DEFAULT_CORE, memory: Path = DEFAULT_MEMORY) -> dict:
    missing = _required_missing(core, memory)
    counts: dict[str, Any] = {"missing_files": missing}
    if not missing:
        references = [_record(row, "reference_archive.csv")
                      for row in _read_csv(core / "reference_archive.csv")]
        candidates = _read_csv(core / "candidate_pool.csv")
        targets = _read_csv(core / "target_response_bank.csv")
        compact = _jsonl(memory / "compact_bank" / "episodes.jsonl")
        primary_inputs = [core / name for name in FINGERPRINT_FILES]
        primary_inputs += [memory / name for name in FINGERPRINT_MEMORY_FILES]
        counts.update({
            "legacy_reference_rows": len(references),
            "legacy_reference_usable_failures": sum(
                row["build_id"] == "idm_ref" and is_usable_outcome(row) and
                row.get("ego_collision") is True for row in references),
            "legacy_reference_passes": sum(
                row["build_id"] == "idm_ref" and is_parent_pass(row) for row in references),
            "legacy_candidate_pool_rows": len(candidates),
            "legacy_target_rows": len(targets),
            "compact_bank_rows": len(compact),
            "compact_rows_by_build": dict(Counter(row.get("build_id") for row in compact)),
            "compact_outcomes_by_build": {
                build: {"usable_failures": sum(
                    row.get("build_id") == build and row.get("ego_collision") is True and
                    is_usable_outcome(row) for row in compact),
                        "parent_passes": sum(row.get("build_id") == build and
                                             is_parent_pass(row) for row in compact)}
                for build in sorted({row.get("build_id") for row in compact})},
            "primary_input_sha256": {path.as_posix(): _sha256(path)
                                     for path in primary_inputs if path.is_file()},
        })
    counts["fingerprinted_file_count"] = len(_input_manifest(core, memory)[0])
    return counts


def _candidate_from_parent(row: dict) -> dict:
    scenario = dict(row["scenario"])
    scenario["context_id"] = row.get("context_id", scenario.get("context_id"))
    return {"scenario_id": row["scenario_id"], "template_id": row["template_id"],
            "scenario": scenario, "context_id": row.get("context_id"),
            "parent_pass": is_parent_pass(row), "completed": row.get("completed"),
            "ego_collision": row.get("ego_collision"), "min_ttc": row.get("min_ttc"),
            "min_clearance": row.get("min_clearance")}


def _task_status(history: list[dict], pool_failures: int | None,
                 missing_cache: bool = False) -> list[str]:
    if missing_cache:
        return ["MISSING_CACHE"]
    result = []
    if not history:
        result.append("NO_COMPATIBLE_HISTORY")
    elif history_kind(history) == "pass_only":
        result.append("PASS_ONLY_HISTORY")
    elif any(row.get("ego_collision") is True and is_usable_outcome(row) for row in history):
        result.append("HAS_FAILURE_MEMORY")
    if pool_failures == 0:
        result.append("NO_TARGET_FAILURE_IN_POOL")
    return result


def _task_input(task_id: str, method: str, repeat: int, history_before: list[dict],
                history: list[dict], candidates: list[dict], cards: list,
                method_history: list[dict], source_build_ids: list[str],
                target_failures: int | None, missing: list[str] | None = None) -> dict:
    failures = sum(row.get("ego_collision") is True and is_usable_outcome(row)
                   for row in history)
    passes = sum(is_parent_pass(row) for row in history)
    contexts = sorted({row.get("context_id", row.get("scenario", {}).get(
        "context_id", "legacy_unspecified")) for row in history})
    return {
        "task_id": task_id, "method": method, "repeat": repeat,
        "selection_visible": {
            "history_record_count": len(history), "history_pass_count": passes,
            "history_failure_count": failures,
            "history_build_ids": sorted({row.get("build_id", "unknown") for row in history}),
            "history_context_ids": contexts,
            "source_history_count": len(method_history),
            "source_history_failure_count": sum(
                row.get("ego_collision") is True and is_usable_outcome(row)
                for row in method_history),
            "source_build_ids": source_build_ids,
            "initial_pattern_count": len(cards),
            "history_kind": history_kind(history),
            "excluded_record_counts_by_reason": excluded_history_counts(history_before,
                                                                         candidates),
            "history_record_count_before_context": len(history_before),
        },
        "offline_evaluator": {
            "target_failure_pool_count": target_failures,
            "has_target_regression": (target_failures > 0 if target_failures is not None else None),
            "oracle_failure_count_at_B": (min(REPLAY_BUDGET, target_failures)
                                          if target_failures is not None else None),
        },
        "missing_cache_files": missing or [],
    }


def _summaries(task_id: str, target_build: str, method: str, repeat: int,
               queries: list[dict], target_failures: int, candidate_count: int,
               history_rows: list[dict], initial_patterns: int) -> list[dict]:
    def discovered(query: dict) -> bool:
        if query.get("mode") == "regression":
            return query.get("regression") is True
        return query.get("valid_collision") is True

    horizon = len(queries)
    first_rank = next((int(row["rank"]) for row in queries if discovered(row)), None)
    history_failures = sum(row.get("ego_collision") is True and is_usable_outcome(row)
                           for row in history_rows)
    history_passes = sum(is_parent_pass(row) for row in history_rows)
    hist_kind = history_kind(history_rows)
    base_status = _task_status(history_rows, target_failures)
    result = []
    for checkpoint in CHECKPOINTS:
        n = min(checkpoint, horizon)
        prefix = queries[:n]
        count = sum(discovered(row) for row in prefix)
        status = list(base_status)
        if target_failures > 0 and checkpoint == REPLAY_BUDGET and count >= target_failures:
            status.append("BASELINE_AT_ORACLE_CEILING")
        result.append({
            "task_id": task_id, "target_build": target_build, "method": method,
            "repeat": repeat, "budget": checkpoint, "actual_queries": n,
            "candidate_count": candidate_count, "failure_pool_count": target_failures,
            "failure_count": count, "first_failure_rank": first_rank,
            "first_failure_censored": (">B" if first_rank is None else first_rank),
            "failure_recall": count / target_failures if target_failures else None,
            "history_record_count": len(history_rows),
            "initial_history_pass_count": history_passes,
            "initial_history_failure_count": history_failures,
            "initial_pattern_count": initial_patterns,
            "history_kind": hist_kind,
            "status": "|".join(status) if status else "REPLAYED",
        })
    return result


def _write_session(output: Path, task_id: str, target_build: str, method: str,
                   repeat: int, mode: str, snapshot_before: str,
                   candidates: list[dict], run_result: dict,
                   snapshot_after: str | None = None) -> None:
    directory = output / "sessions" / task_id / method / f"repeat_{repeat}"
    _atomic_csv(directory / "queries.csv", run_result["queries"])
    _atomic_jsonl(directory / "updates.jsonl", run_result["updates"])
    session = Session(
        session_id=f"{task_id}:{method}:{repeat}", task_id=task_id,
        target_build_id=target_build, mode=mode,
        history_snapshot_before=snapshot_before,
        candidate_ids=[row["scenario_id"] for row in candidates],
        queried_execution_ids=[row["execution_id"] for row in run_result["queries"]],
        observations=run_result["observations"],
        created_pattern_ids=[row.get("pattern_id") for row in run_result["cards"]
                             if row.get("created_in_session") ==
                             f"{task_id}:{method}:{repeat}"])
    session.history_snapshot_after = snapshot_after or snapshot_before
    _atomic_json(directory / "session.json", session.as_dict())
    _atomic_jsonl(directory / "patterns.jsonl", run_result["cards"])


def _run_selector_task(output: Path, task_id: str, target_build: str,
                       candidates: list[dict], history_before: list[dict],
                       evaluator: dict[str, dict], mode: str,
                       parent_build: str | None, seed_for,
                       snapshot_before: str | None = None,
                       only_method: str | None = None,
                       only_repeat: int | None = None,
                       paired_repeats: int = 1) -> dict:
    history = contextual_history(history_before, candidates)
    history = [dict(row) for row in history if is_usable_outcome(row)]
    total_failures = sum(row.get("ego_collision") is True and not row.get("inconclusive", False)
                         and (mode != "regression" or next(
                             item for item in candidates if item["scenario_id"] == sid).get(
                                 "parent_pass") is True)
                         for sid, row in evaluator.items())
    before_hash = snapshot_before or stable_hash(history_before)
    run_results = []
    method_summaries = []
    task_inputs = []
    fit_count = 0
    for method in ([only_method] if only_method else METHODS):
        repeats = list(range(RANDOM_REPEATS if method == "Random" else paired_repeats))
        if only_repeat is not None:
            repeats = [only_repeat]
        for repeat in repeats:
            method_cards = [] if method == "FBRT-NoMemory" else build_pattern_cards(
                history, session_id=f"{task_id}:{method}:{repeat}")
            oracle = TargetOracle({key: evaluator[key] for key in
                                   {row["scenario_id"] for row in candidates}})
            method_families = {row["build_id"]: row.get("family", _family(row["build_id"]))
                               for row in history}
            run_seed = seed_for(method, repeat)
            queries, observations, learned_cards, updates = run_selector(
                method, candidates, history, oracle, REPLAY_BUDGET, run_seed,
                target_build_id=target_build, parent_build_id=parent_build,
                mode=mode, session_id=f"{task_id}:{method}:{repeat}",
                family_by_build=method_families, initial_cards=method_cards)
            fit_count += sum(int(update.get("model_fit_count", 0))
                             for update in updates if update.get("event") == "model_fit_ledger")
            query_rows = [{**row, "task_id": task_id, "repeat": repeat} for row in queries]
            method_history = history if method != "FBRT-NoMemory" else []
            task_input = _task_input(
                task_id, method, repeat, history_before, history, candidates,
                method_cards, method_history,
                sorted({row["build_id"] for row in method_history}), total_failures)
            run = {"method": method, "repeat": repeat, "seed": run_seed,
                   "queries": query_rows, "observations": observations,
                   "cards": [card.as_dict() if hasattr(card, "as_dict") else card
                             for card in learned_cards],
                   "updates": updates, "task_input": task_input,
                   "snapshot_before": before_hash}
            run_results.append(run)
            method_summaries.extend(_summaries(
                task_id, target_build, method, repeat, queries, total_failures,
                len(candidates), history, len(method_cards)))
    return {"task_id": task_id, "target_build": target_build, "mode": mode,
            "candidates": candidates, "history": history,
            "history_before": history_before, "evaluator": evaluator,
            "run_results": run_results, "summaries": method_summaries,
            "task_inputs": [run["task_input"] for run in run_results],
            "model_fit_count": fit_count,
            "logical_query_count": sum(len(run["queries"]) for run in run_results),
            "total_failures": total_failures}


def _cache_task(output: Path, fingerprint: str, key: str, work) -> dict:
    cache_path = output / "task_cache" / f"{stable_hash(key)[:24]}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("run_fingerprint") == fingerprint:
            return cached["result"]
    result = work()
    _atomic_json(cache_path, {"run_fingerprint": fingerprint, "result": result})
    return result


def _write_task_outputs(output: Path, task: dict) -> None:
    for run in task["run_results"]:
        _write_session(output, task["task_id"], task["target_build"],
                       run["method"], run["repeat"], task["mode"],
                       run["snapshot_before"], task["candidates"], run,
                       run.get("snapshot_after") or task.get("snapshot_after"))


def _missing_summary(task_id: str, target: str, mode: str,
                     candidates: list[dict], missing: list[str],
                     status: str = "MISSING_CACHE") -> dict:
    return {"task_id": task_id, "target_build": target, "mode": mode,
            "status": status, "candidate_count": len(candidates),
            "missing_cache_files": missing}


def _load_legacy(core: Path) -> tuple[list[dict], list[dict], dict[str, dict[str, dict]]]:
    reference = [_record(row, "reference_archive.csv")
                 for row in _read_csv(core / "reference_archive.csv")]
    parent = {row["scenario_id"]: row for row in reference if row["build_id"] == "idm_ref"}
    candidates = []
    for row in _read_csv(core / "candidate_pool.csv"):
        parent_row = parent.get(row["scenario_id"])
        if parent_row is None:
            continue
        candidate = _candidate_from_parent(parent_row)
        if candidate["parent_pass"] is True:
            candidates.append(candidate)
    targets: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in _read_csv(core / "target_response_bank.csv"):
        record = _record(row, "target_response_bank.csv")
        record["visibility"] = "evaluator_only"
        targets[record["build_id"]][record["scenario_id"]] = record
    return reference, candidates, targets


def _legacy_tasks(output: Path, core: Path, fingerprint: str,
                  paired_repeats: int = 1) -> list[dict]:
    reference, candidates_all, target_banks = _load_legacy(core)
    task_results = []
    for seed in sorted({int(row["scenario_id"].split(":", 1)[0]) for row in candidates_all}):
        candidates = [item for item in candidates_all
                      if int(item["scenario_id"].split(":", 1)[0]) == seed]
        history = [row for row in reference if row.get("build_id") == "idm_ref"
                   and row.get("simulator_seed") == seed
                   and row.get("visibility") == "historical"]
        for target in LEGACY_TARGETS:
            task_id = f"legacy_regression_seed{seed}_{target}"
            target_bank = target_banks.get(target, {})
            missing = sorted(set(row["scenario_id"] for row in candidates) - set(target_bank))
            if not candidates:
                task_results.append(_missing_summary(task_id, target, "regression",
                                                     candidates, ["no parent-pass candidates"],
                                                     status="NO_ELIGIBLE_CASES"))
                continue
            if missing:
                task_results.append(_missing_summary(task_id, target, "regression",
                                                     candidates, missing))
                continue
            evaluator = {row["scenario_id"]: target_bank[row["scenario_id"]]
                         for row in candidates}
            run_fingerprint = fingerprint
            result = _cache_task(output, run_fingerprint, task_id, lambda: _run_selector_task(
                output, task_id, target, candidates, history, evaluator, "regression", "idm_ref",
                lambda method, repeat: _regression_seed(task_id, method, repeat,
                                                        paired_repeats),
                paired_repeats=paired_repeats))
            _write_task_outputs(output, result)
            task_results.append(result)
    return task_results


def _load_compact(memory: Path) -> tuple[list[dict], dict[str, dict[str, dict]]]:
    cases = _jsonl(memory / "compact_bank" / "scenario_cases.jsonl")
    bank_rows = _jsonl(memory / "compact_bank" / "episodes.jsonl")
    candidates_by_id = {}
    for case in cases:
        scenario = dict(case)
        scenario["context_id"] = case.get("context_id", "legacy_unspecified")
        candidates_by_id[case["scenario_id"]] = {
            "scenario_id": case["scenario_id"], "template_id": case["template_id"],
            "scenario": scenario, "context_id": case.get("context_id"),
        }
    bank: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in bank_rows:
        bank[row["build_id"]][row["scenario_id"]] = dict(row)
        bank[row["build_id"]][row["scenario_id"]]["family"] = _family(row["build_id"])
    return list(candidates_by_id.values()), bank


def _compact_regressions(output: Path, memory: Path, fingerprint: str,
                         paired_repeats: int = 1) -> list[dict]:
    candidates_all, banks = _load_compact(memory)
    results = []
    for parent_build, target_build, task_id in COMPACT_BUILD_PAIRS:
        parent_bank = banks.get(parent_build, {})
        candidates = []
        for candidate in candidates_all:
            row = parent_bank.get(candidate["scenario_id"])
            if row and is_parent_pass(row):
                candidates.append({**candidate, "parent_pass": True,
                                   "completed": True, "ego_collision": False})
        target_bank = banks.get(target_build, {})
        missing = sorted(set(row["scenario_id"] for row in candidates) - set(target_bank))
        if not parent_bank:
            missing.append(f"compact bank build absent: {parent_build}")
        if not parent_bank:
            results.append(_missing_summary(task_id, target_build, "regression",
                                            candidates, missing or [f"compact bank build absent: {parent_build}"]))
            continue
        if not candidates:
            results.append(_missing_summary(task_id, target_build, "regression",
                                            candidates, ["no parent-pass candidates"],
                                            status="NO_ELIGIBLE_CASES"))
            continue
        if missing:
            results.append(_missing_summary(task_id, target_build, "regression",
                                            candidates, missing))
            continue
        history = [{**row, "visibility": "historical", "family": _family(parent_build)}
                   for row in parent_bank.values()]
        evaluator = {row["scenario_id"]: target_bank[row["scenario_id"]]
                     for row in candidates}
        result = _cache_task(output, fingerprint, task_id, lambda: _run_selector_task(
            output, task_id, target_build, candidates, history, evaluator,
            "regression", parent_build,
            lambda method, repeat: _regression_seed(task_id, method, repeat,
                                                    paired_repeats),
            paired_repeats=paired_repeats))
        _write_task_outputs(output, result)
        results.append(result)
    return results


def _initial_cross_agent_history(memory: Path, bank_rows: list[dict]) -> list[dict]:
    bank_ids = {row.get("execution_id") for row in bank_rows}
    records = _jsonl(memory / "archive_v2.jsonl")
    return [dict(row) for row in records
            if row.get("execution_id") not in bank_ids
            and row.get("visibility") != "evaluator_only"
            and (row.get("visibility") == "historical" or
                 (row.get("visibility") is None and
                  row.get("source_file") == "reference_archive.csv"))]


def _cross_pair(output: Path, fingerprint: str, task_id: str,
                target_build: str, candidates: list[dict], history: list[dict],
                evaluator: dict[str, dict], method: str, seed: int, repeat: int,
                task_mode: str) -> dict:
    result = _run_selector_task(
        output, task_id, target_build, candidates, history, evaluator,
        task_mode, None,
        lambda method, _repeat: seed + repeat,
        snapshot_before=_snapshot_hash(history), only_method=method, only_repeat=repeat)
    return result


def _compact_cross_agent(output: Path, memory: Path, fingerprint: str) -> list[dict]:
    candidates, banks = _load_compact(memory)
    bank_rows = _jsonl(memory / "compact_bank" / "episodes.jsonl")
    base_records = _initial_cross_agent_history(memory, bank_rows)
    mobil_bank = banks.get("mobil_ref_v2", {})
    ppo_bank = banks.get("ppo_ref_v2", {})
    required_ids = {row["scenario_id"] for row in candidates}
    results = []
    for method in METHODS:
        repeats = RANDOM_REPEATS if method == "Random" else 1
        for repeat in range(repeats):
            pair_key = f"cross:{method}:{repeat}"

            def work():
                first_id = f"cross_agent_{method}_mobil_to_next"
                second_id = f"cross_agent_{method}_ppo_ref_after_mobil"
                first_missing = sorted(required_ids - set(mobil_bank))
                pair_result = {"pair_key": pair_key, "run_results": [],
                               "summaries": [], "task_inputs": [],
                               "model_fit_count": 0, "logical_query_count": 0,
                               "missing_statuses": []}
                if first_missing:
                    pair_result["missing_statuses"].append(
                        _missing_summary(first_id, "mobil_ref_v2", "cross_agent",
                                         candidates, first_missing))
                    pair_result["missing_statuses"].append(
                        _missing_summary(second_id, "ppo_ref_v2", "cross_agent",
                                         candidates, ["first-agent cache incomplete"]))
                    return pair_result
                first_eval = {sid: mobil_bank[sid] for sid in required_ids}
                first = _cross_pair(output, fingerprint, first_id, "mobil_ref_v2",
                                    candidates, base_records, first_eval, method,
                                    99001, repeat, "cross_agent")
                pair_result["run_results"].append(first)
                pair_result["summaries"].extend(first["summaries"])
                pair_result["task_inputs"].extend(first["task_inputs"])
                pair_result["model_fit_count"] += first["model_fit_count"]
                pair_result["logical_query_count"] += first["logical_query_count"]

                first_memory = [dict(row, family=_family("mobil_ref_v2"))
                                for run in first["run_results"] if run["method"] == method
                                and run["repeat"] == repeat for row in run["observations"]]
                branch = {row["execution_id"]: row for row in base_records}
                branch.update((row["execution_id"], row) for row in first_memory)
                branch_history = list(branch.values())
                branch_dir = output / "sessions" / "compact_cross_agent" / "branches" / \
                    method / f"repeat_{repeat}" / "history_branch"
                before_hash = _snapshot_hash(base_records)
                after_hash = _snapshot_hash(branch_history)
                first["snapshot_after"] = after_hash
                _atomic_jsonl(branch_dir / "archive_v2.jsonl",
                              sorted(branch_history, key=lambda row: row["execution_id"]))
                _atomic_json(branch_dir / "history_snapshots" / f"{before_hash}.json",
                             {"snapshot_hash": before_hash, "record_count": len(base_records),
                              "source": "frozen_initial_archive"})
                _atomic_json(branch_dir / "history_snapshots" / f"{after_hash}.json",
                             {"snapshot_hash": after_hash, "record_count": len(branch_history),
                              "source": "queried_first_agent_observations"})

                ppo_missing = sorted(required_ids - set(ppo_bank))
                if ppo_missing:
                    pair_result["missing_statuses"].append(
                        _missing_summary(second_id, "ppo_ref_v2", "cross_agent",
                                         candidates, ppo_missing))
                    return pair_result
                second_eval = {sid: ppo_bank[sid] for sid in required_ids}
                second = _cross_pair(output, fingerprint, second_id, "ppo_ref_v2",
                                     candidates, branch_history, second_eval, method,
                                     99002, repeat, "cross_agent")
                second["snapshot_before"] = after_hash
                for run in second["run_results"]:
                    run["snapshot_before"] = after_hash
                pair_result["run_results"].append(second)
                pair_result["summaries"].extend(second["summaries"])
                pair_result["task_inputs"].extend(second["task_inputs"])
                pair_result["model_fit_count"] += second["model_fit_count"]
                pair_result["logical_query_count"] += second["logical_query_count"]
                return pair_result

            pair = _cache_task(output, fingerprint, pair_key, work)
            if pair["run_results"]:
                first = pair["run_results"][0]
                first_method_run = next((run for run in first["run_results"]
                                         if run["method"] == method and
                                         run["repeat"] == repeat), None)
                first_observations = (first_method_run or {}).get("observations", [])
                branch = {row["execution_id"]: row for row in base_records}
                branch.update((row["execution_id"], dict(row, family=_family("mobil_ref_v2")))
                              for row in first_observations)
                branch_rows = list(branch.values())
                after_hash = _snapshot_hash(branch_rows)
                first["snapshot_after"] = after_hash
                branch_dir = output / "sessions" / "compact_cross_agent" / "branches" / \
                    method / f"repeat_{repeat}" / "history_branch"
                _atomic_jsonl(branch_dir / "archive_v2.jsonl",
                              sorted(branch_rows, key=lambda row: row["execution_id"]))
                for later in pair["run_results"][1:]:
                    later["snapshot_before"] = after_hash
                    for run in later["run_results"]:
                        run["snapshot_before"] = after_hash
            for task in pair["run_results"]:
                _write_task_outputs(output, task)
            results.append(pair)
    return results


def _missing_cross_tasks(missing: list[str], candidates: list[dict] | None = None) -> list[dict]:
    candidates = candidates or []
    rows = []
    for method in METHODS:
        for repeat in range(RANDOM_REPEATS if method == "Random" else 1):
            rows.append(_missing_summary(
                f"cross_agent_{method}_mobil_to_next", "mobil_ref_v2", "cross_agent",
                candidates, missing))
            rows.append(_missing_summary(
                f"cross_agent_{method}_ppo_ref_after_mobil", "ppo_ref_v2", "cross_agent",
                candidates, missing))
    return rows


def _flatten_results(groups: list[dict]) -> tuple[list[dict], list[dict], list[dict], int, int]:
    summaries, inputs, statuses = [], [], []
    fit_count = query_count = 0
    for group in groups:
        if "run_results" not in group:
            statuses.append(group)
            continue
        summaries.extend(group.get("summaries", []))
        inputs.extend(group.get("task_inputs", []))
        fit_count += int(group.get("model_fit_count", 0))
        query_count += int(group.get("logical_query_count", 0))
        statuses.extend(group.get("missing_statuses", []))
    return summaries, inputs, statuses, fit_count, query_count


def _before_after(memory: Path, new_summary: list[dict]) -> list[dict]:
    old_path = memory / "summary_by_task.csv"
    if not old_path.is_file():
        return [{"task_id": row["task_id"], "method": row["method"],
                 "repeat": row["repeat"], "budget": row["budget"],
                 "comparison_status": "before_missing",
                 "failure_count_after": row.get("failure_count"),
                 "failure_pool_count_after": row.get("failure_pool_count")}
                for row in new_summary if int(row.get("budget", 0)) == REPLAY_BUDGET]
    old_rows = _read_csv(old_path)
    old = {}
    for row in old_rows:
        try:
            key = (row.get("task_id"), row.get("method"), int(row.get("repeat", 0)),
                   int(row.get("budget", 0)))
        except (TypeError, ValueError):
            continue
        old[key] = row
    comparison = []
    for row in new_summary:
        if int(row.get("budget", 0)) != REPLAY_BUDGET:
            continue
        key = (row.get("task_id"), row.get("method"), int(row.get("repeat", 0)),
               int(row.get("budget", 0)))
        prior = old.get(key)
        if prior is None:
            comparison.append({"task_id": key[0], "method": key[1], "repeat": key[2],
                               "budget": key[3], "comparison_status": "before_missing",
                               "failure_count_after": row.get("failure_count"),
                               "failure_pool_count_after": row.get("failure_pool_count")})
            continue
        try:
            before_count = float(prior.get("failure_count", ""))
            after_count = float(row.get("failure_count", ""))
        except (TypeError, ValueError):
            before_count = after_count = None
        comparison.append({"task_id": key[0], "method": key[1], "repeat": key[2],
                           "budget": key[3], "comparison_status": "compared",
                           "failure_count_before": before_count,
                           "failure_count_after": after_count,
                           "failure_count_delta": (after_count - before_count
                                                   if before_count is not None and
                                                   after_count is not None else None),
                           "failure_pool_count_before": prior.get("failure_pool_count"),
                           "failure_pool_count_after": row.get("failure_pool_count")})
    return comparison


def _empirical_effect(summary: list[dict]) -> tuple[str, list[dict]]:
    at_budget = [row for row in summary if int(row.get("budget", 0)) == REPLAY_BUDGET]
    def comparable_task(task_id: str) -> str:
        for method in METHODS:
            prefix = f"cross_agent_{method}_"
            if task_id.startswith(prefix):
                return "cross_agent_" + task_id[len(prefix):]
        return task_id

    tasks = sorted({comparable_task(row.get("task_id", "")) for row in at_budget})
    comparisons = []
    gains, losses = 0, 0
    for task_id in tasks:
        local = [row for row in at_budget
                 if comparable_task(row.get("task_id", "")) == task_id]
        memory_values = [float(row["failure_count"]) for row in local
                         if row.get("method") == "FBRT-Memory" and
                         row.get("failure_count") is not None]
        no_memory_values = [float(row["failure_count"]) for row in local
                            if row.get("method") == "FBRT-NoMemory" and
                            row.get("failure_count") is not None]
        pool = next((int(row["failure_pool_count"]) for row in local
                     if row.get("failure_pool_count") is not None), 0)
        if pool <= 0 or not memory_values:
            comparisons.append({"task_id": task_id, "status": "not_applicable",
                                "target_failure_pool_count": pool})
            continue
        memory_mean = sum(memory_values) / len(memory_values)
        baseline_means = {}
        for baseline in ("FBRT-NoMemory", "HistoryRank-UCB-v2",
                         "FailureDistance-v2", "Random"):
            values = [float(row["failure_count"]) for row in local
                      if row.get("method") == baseline and row.get("failure_count") is not None]
            if values:
                baseline_means[baseline] = sum(values) / len(values)
        no_memory_mean = baseline_means.get("FBRT-NoMemory")
        if no_memory_mean is not None:
            delta = memory_mean - no_memory_mean
            gains += int(delta > 0)
            losses += int(delta < 0)
        comparisons.append({"task_id": task_id, "status": "compared",
                            "memory_failure_count_mean": memory_mean,
                            "baseline_failure_count_means": baseline_means,
                            "delta_vs_no_memory": (memory_mean - no_memory_mean
                                                   if no_memory_mean is not None else None),
                            "target_failure_pool_count": pool})
    comparable = gains + losses + sum(row.get("delta_vs_no_memory") == 0
                                      for row in comparisons
                                      if row.get("delta_vs_no_memory") is not None)
    if comparable == 0:
        effect = "not_applicable"
    elif gains and losses:
        effect = "mixed"
    elif gains:
        effect = "gain"
    else:
        effect = "no_gain"
    return effect, comparisons


def _collision_decomposition(memory: Path) -> dict:
    rows = [row for row in _jsonl(memory / "compact_bank" / "episodes.jsonl")
            if row.get("template_id") == "fbrt_cutout_static"
            and row.get("ego_collision") is True and not row.get("inconclusive", False)]
    roles = []
    for row in rows:
        role = row.get("collision_partner_role") or row.get("collision_partner") or "unknown"
        role = str(role).lower()
        roles.append(role if role in {"lead", "static", "rear"} else "unknown")
    counts = Counter(roles)
    return {role: counts.get(role, 0) for role in ("lead", "static", "rear", "unknown")}


def _s02_unreliable_exit_count(memory: Path) -> int:
    return sum(row.get("template_id") == "fbrt_cutout_static"
               and row.get("event_times", {}).get("first_exit_s") == 0.0
               for row in _jsonl(memory / "compact_bank" / "episodes.jsonl"))


def _make_report(report_path: Path, manifest: dict, summaries: list[dict],
                 statuses: list[dict], effect: str, comparisons: list[dict],
                 s02: dict, s02_zero_exit_count: int, test_result: str) -> None:
    engineering = "fixed_with_missing_cache" if any(
        row.get("status") == "MISSING_CACHE" for row in statuses) else "fixed_and_replayed"
    task_pairs = {(row.get("task_id"), row.get("method"), row.get("repeat"))
                  for row in summaries}
    count_rows = [row for row in summaries if int(row.get("budget", 0)) == REPLAY_BUDGET]

    def display_task(task_id: str) -> str:
        for method in METHODS:
            prefix = f"cross_agent_{method}_"
            if task_id.startswith(prefix):
                return "cross_agent_" + task_id[len(prefix):]
        return task_id

    grouped_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in count_rows:
        grouped_rows[(display_task(row.get("task_id", "")), row.get("method", ""))].append(row)
    lines = [
        "# FBRT Memory repair replay report", "",
        f"- `engineering_status`: **{engineering}**",
        f"- `empirical_effect`: **{effect}**",
        "- Additional physical episodes: **0**; policy training runs: **0**.",
        f"- Offline task runs recorded: **{len(task_pairs)}**; result tasks: **{len({r.get('task_id') for r in count_rows})}**.",
        f"- Unit test result: `{test_result}`.",
        "- The frozen input fingerprints matched before and after replay: "
        f"`{manifest.get('input_bank_fingerprints_unchanged')}`.", "",
        "## Code changes", "",
        "- Historical modeling now keeps valid collision terminated outcomes and excludes evaluator only or incomplete rows; regression candidates still require a complete parent pass.",
        "- RBF schemas include bias, named coordinates, failure centers, coverage centers, template/context/parameterization identity, and feature semantics. Source priors are reindexed by feature ID; new features receive N(0, 4).",
        "- UCB uses regression rewards for regression tasks and valid collision rewards for cross agent tasks. Inconclusive outcomes count as queries without positive reward.",
        "- Observed outcomes and pattern cards retain collision partner, event, contract, signature, and trajectory fields when present. Missing partner labels remain unknown.",
        "- The offline replay reads the frozen banks and writes only under this repair output directory.", "",
        "## Task effects at B=20", "",
        "| Task | Method | Queries | Failures found | Failure pool | Recall | History state |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for (task_id, method), rows in sorted(grouped_rows.items()):
        average = lambda key: sum(float(row[key]) for row in rows
                                  if row.get(key) not in (None, "")) / max(
                                      1, sum(row.get(key) not in (None, "") for row in rows))
        recalls = [float(row["failure_recall"]) for row in rows
                   if row.get("failure_recall") not in (None, "", "None", "NA")]
        recall_text = f"{sum(recalls) / len(recalls):.3f}" if recalls else "NA"
        method_text = f"Random (mean, n={len(rows)})" if method == "Random" else method
        failure_pool = rows[0].get("failure_pool_count")
        lines.append(f"| {task_id} | {method_text} | {average('actual_queries'):.1f} | "
                     f"{average('failure_count'):.2f} | {failure_pool} | {recall_text} | "
                     f"{rows[0].get('history_kind')} |")
    lines.extend(["", "## Applicability and interpretation", "",
                  "`HAS_FAILURE_MEMORY`, `PASS_ONLY_HISTORY`, and `NO_COMPATIBLE_HISTORY` describe the source view actually supplied to each run. `NO_TARGET_FAILURE_IN_POOL` means recall is not applicable. `BASELINE_AT_ORACLE_CEILING` marks runs whose B=20 discoveries equal the complete target failure pool. Missing files are listed as `MISSING_CACHE`; no missing episode was generated.",
                  "",
                  "At B=20, Memory versus NoMemory and the existing baselines are:",
                  "", "| Task | Failure pool | Memory | NoMemory | Δ vs NoMemory | HistoryRank-UCB | FailureDistance | Random mean |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        baselines = row.get("baseline_failure_count_means", {})
        if row.get("status") != "compared":
            lines.append(f"| {row.get('task_id')} | {row.get('target_failure_pool_count')} | NA | NA | NA | NA | NA | NA |")
            continue
        def cell(value):
            return "NA" if value is None else f"{float(value):.2f}"
        lines.append(f"| {row.get('task_id')} | {row.get('target_failure_pool_count')} | "
                     f"{cell(row.get('memory_failure_count_mean'))} | "
                     f"{cell(baselines.get('FBRT-NoMemory'))} | "
                     f"{cell(row.get('delta_vs_no_memory'))} | "
                     f"{cell(baselines.get('HistoryRank-UCB-v2'))} | "
                     f"{cell(baselines.get('FailureDistance-v2'))} | "
                     f"{cell(baselines.get('Random'))} |")
    lines.extend(["", "Random entries are means across the original ten repeats; other methods use their single original run.",
                  f"Compact S02 observed collision partner counts (lead/static/rear/unknown): `{json.dumps(s02, sort_keys=True)}`.",
                  f"The original S02 `first_exit_s=0` values are retained as unreliable event data ({s02_zero_exit_count} bank rows); no v4 trajectory or inferred object label changed the v2 ranking labels.",
                  "", "## Answers", "",
                  "1. Historical failures reach pattern construction and source fitting when the task context admits them; the task input ledger records their exact counts and IDs.",
                  "2. Feature coefficients and variances follow stable IDs after each dynamic center insertion; source prior arrays are refit against all queried valid observations exactly once per posterior fit.",
                  "3. UCB rewards follow the task mode, and query/update fields record the valid collision and reward used.",
                  f"4. The measured Memory effect is `{effect}` under the existing tasks; task level deltas are in `comparison_before_after.csv` and `memory_vs_baselines.json`.",
                  "5. Tasks without source failures, compatible history, target regressions, or complete cache coverage are labeled separately; they do not support the corresponding migration claim.",
                  "", "## Provenance", "",
                  f"- Source HEAD: `{manifest.get('source_git_head')}`",
                  f"- Replay fingerprint: `{manifest.get('run_fingerprint')}`",
                  f"- Logical queries: {manifest.get('logical_query_count')}",
                  f"- Model fits: {manifest.get('model_fit_count')}",
                  f"- Input files fingerprinted: {len(manifest.get('input_sha256', {}))}",
                  f"- Missing inputs: `{json.dumps(manifest.get('missing_inputs', []), ensure_ascii=False)}`",
                  "- See `manifest.json` and `task_inputs.jsonl` for detailed provenance.", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def replay_bank(output: Path = DEFAULT_OUTPUT, core: Path = DEFAULT_CORE,
                memory: Path = DEFAULT_MEMORY, test_result: str = "not recorded",
                paired_repeats: int = 1) -> dict:
    if paired_repeats < 1:
        raise ValueError("paired_repeats must be positive")
    output = output.resolve()
    core, memory = core.resolve(), memory.resolve()
    if output == core or output == memory or core in output.parents or memory in output.parents:
        raise ValueError("repair output must be separate from the frozen input banks")
    before_hashes, input_missing = _input_manifest(core, memory)
    missing = _required_missing(core, memory)
    config = {"budget": REPLAY_BUDGET, "methods": list(METHODS),
              "random_repeats": RANDOM_REPEATS, "legacy_targets": list(LEGACY_TARGETS),
              "paired_regression_repeats": paired_repeats,
              "compact_build_pairs": [list(item) for item in COMPACT_BUILD_PAIRS],
              "input_roots": {"core": core.as_posix(), "memory": memory.as_posix()},
              "additional_physical_episodes": 0, "policy_training_runs": 0}
    code_hashes = {path: _sha256(Path(path)) for path in CODE_FILES if Path(path).is_file()}
    base_manifest = {"schema_version": "fbrt-repair-replay-v1",
                     "source_git_head": _git_head(), "input_sha256": before_hashes,
                     "missing_inputs": missing, "config": config,
                     "source_code_sha256": code_hashes,
                     "python_version": sys.version}
    fingerprint = stable_hash(base_manifest)
    existing_manifest = output / "manifest.json"
    if existing_manifest.is_file():
        prior = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") != fingerprint:
            raise ValueError("existing repair output belongs to different code, inputs, or configuration")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("repair output exists without a manifest; refusing to mix result versions")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**base_manifest, "run_fingerprint": fingerprint,
                "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_bank_fingerprints_unchanged": False,
                "additional_physical_episodes": 0, "policy_training_runs": 0,
                "completed_replay_tasks": 0, "missing_cache_tasks": [],
                "logical_query_count": 0, "model_fit_count": 0}
    _atomic_json(existing_manifest, manifest)

    core_required = [core / name for name in FINGERPRINT_FILES]
    core_missing = [path.as_posix() for path in core_required if not path.is_file()]
    if not core_missing:
        legacy_results = _legacy_tasks(output, core, fingerprint, paired_repeats)
    else:
        legacy_results = [_missing_summary("legacy_regression", "multiple", "regression",
                                           [], core_missing)]
    compact_required = [memory / "compact_bank" / "scenario_cases.jsonl",
                        memory / "compact_bank" / "episodes.jsonl"]
    compact_missing = [path.as_posix() for path in compact_required if not path.is_file()]
    if not compact_missing:
        compact_results = _compact_regressions(output, memory, fingerprint,
                                               paired_repeats)
    else:
        compact_results = [_missing_summary(task_id, target, "regression", [], compact_missing)
                           for _parent, target, task_id in COMPACT_BUILD_PAIRS]
    if not compact_missing and (memory / "archive_v2.jsonl").is_file():
        cross_results = _compact_cross_agent(output, memory, fingerprint)
    else:
        cross_missing = compact_missing or [(memory / "archive_v2.jsonl").as_posix()]
        cross_candidates = []
        cases_path = memory / "compact_bank" / "scenario_cases.jsonl"
        if cases_path.is_file():
            cross_candidates = _load_compact(memory)[0]
        cross_results = _missing_cross_tasks(cross_missing, cross_candidates)

    all_groups = legacy_results + compact_results + cross_results
    summaries, task_inputs, status_rows, fit_count, query_count = _flatten_results(all_groups)
    comparison = _before_after(memory, summaries)
    effect, method_comparisons = _empirical_effect(summaries)
    after_hashes, after_missing = _input_manifest(core, memory)
    inputs_unchanged = before_hashes == after_hashes and missing == _required_missing(core, memory)
    manifest.update({"finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "input_sha256_after": after_hashes,
                     "input_bank_fingerprints_unchanged": inputs_unchanged,
                     "input_fingerprint_mismatch": sorted(
                         key for key in set(before_hashes) | set(after_hashes)
                         if before_hashes.get(key) != after_hashes.get(key)),
                     "input_missing_after": after_missing,
                     "completed_replay_tasks": len({row.get("task_id") for row in summaries}),
                     "missing_cache_tasks": [row for row in status_rows
                                             if row.get("status") == "MISSING_CACHE"],
                     "logical_query_count": query_count,
                     "model_fit_count": fit_count,
                     "empirical_effect": effect,
                     "engineering_status": "fixed_with_missing_cache" if any(
                         row.get("status") == "MISSING_CACHE" for row in status_rows)
                         else "fixed_and_replayed"})
    _atomic_json(existing_manifest, manifest)
    _atomic_jsonl(output / "task_inputs.jsonl", task_inputs)
    _atomic_csv(output / "summary_by_task.csv", summaries)
    _atomic_csv(output / "comparison_before_after.csv", comparison)
    _atomic_jsonl(output / "task_status.jsonl", status_rows)
    _atomic_json(output / "memory_vs_baselines.json", method_comparisons)
    _atomic_json(output / "cache_ledger.json", {
        "additional_physical_episodes": 0, "policy_training_runs": 0,
        "input_bank_fingerprints_unchanged": inputs_unchanged,
        "completed_replay_tasks": manifest["completed_replay_tasks"],
        "missing_cache_tasks": manifest["missing_cache_tasks"],
        "logical_query_count": query_count, "model_fit_count": fit_count,
    })
    _make_report(output / "repair_report.md", manifest, summaries, status_rows,
                 effect, method_comparisons, _collision_decomposition(memory),
                 _s02_unreliable_exit_count(memory), test_result)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay FBRT measured banks without simulation")
    parser.add_argument("--offline-only", action="store_true",
                        help="required safety flag; never creates or runs episodes")
    parser.add_argument("--check-inputs", action="store_true",
                        help="only fingerprint inputs and report available history counts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--test-result", default="not recorded")
    parser.add_argument("--paired-repeats", type=int, default=1,
                        help="independent regression repeats with common seeds across methods")
    args = parser.parse_args(argv)
    if args.check_inputs:
        print(json.dumps(check_inputs(args.core_dir, args.memory_dir),
                         ensure_ascii=False, indent=2))
        return 0
    if not args.offline_only:
        parser.error("--offline-only is required to run a replay")
    result = replay_bank(args.output, args.core_dir, args.memory_dir, args.test_result,
                         args.paired_repeats)
    print(json.dumps({key: result[key] for key in (
        "run_fingerprint", "completed_replay_tasks", "logical_query_count",
        "model_fit_count", "missing_cache_tasks", "input_bank_fingerprints_unchanged",
        "engineering_status", "empirical_effect")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
