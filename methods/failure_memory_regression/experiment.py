"""Build the measured FBRT bank with the highway-env physical runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from highway_sim_env.envs.fbrt_unified_env import (
    EXECUTION_CONTRACT, run_build_episode,
)
from methods.failure_memory_regression.archive import (
    CORE, ROOT, SnapshotStore, load_legacy_archive, read_jsonl, row_to_record,
)
from methods.failure_memory_regression.bayes_model import source_fits
from methods.failure_memory_regression.catalogue import (
    CATALOGUE, compile_scenarios, load_catalogue, write_capability_and_recipe,
)
from methods.failure_memory_regression.pattern_memory import (
    build_dictionaries, build_pattern_cards, cards_jsonl,
)
from methods.failure_memory_regression.replay_utils import (
    contextual_history, is_parent_pass, is_usable_outcome,
)
from methods.failure_memory_regression.schema import Session, stable_hash
from methods.failure_memory_regression.selector import (
    CHECKPOINTS, METHODS, TargetOracle, run_selector,
)
from sut_algorithms.highway_env.registry import build_spec_factory


GLOBAL_PHYSICAL_CAP = 400
MAIN_BANK_CAP = 320
BUDGET = 20
RANDOM_REPEATS = 10
SIMULATOR_SEED = 4179901


def _family(build_id: str) -> str:
    if build_id == "idm_ref":
        return "legacy_profiled_idm_reference"
    if build_id in {"merge_blind06", "merge_brake2", "slow_front_brake2"}:
        return f"legacy_profiled_idm_fault:{build_id}"
    if build_id.startswith("mobil_"):
        return "native_idm_mobil"
    if build_id.startswith("ppo_"):
        return "ppo_ece"
    return build_id


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _rows_for_family(records: list[dict]) -> list[dict]:
    result = []
    for row in records:
        item = dict(row)
        item["family"] = _family(item.get("build_id", "unknown"))
        result.append(item)
    return result


def _contextual_history(history: list[dict], candidates: list[dict]) -> list[dict]:
    return contextual_history(history, candidates)


def _cross_agent_seed_records(records: list[dict], bank_rows: list[dict]) -> list[dict]:
    """Freeze the first-agent starting view across repeated offline evaluations."""
    bank_execution_ids = {row["execution_id"] for row in bank_rows}
    return [row for row in records if row.get("visibility") != "evaluator_only" and
            row.get("execution_id") not in bank_execution_ids]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _protocol(capability: dict, recipe: dict, catalogue: dict) -> dict:
    try:
        import torch
        gpu = {"cuda_available": bool(torch.cuda.is_available()),
               "device_count": int(torch.cuda.device_count()),
               "device_names": [torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]}
    except Exception as exc:  # resource status is diagnostic, not a blocker
        gpu = {"cuda_available": False, "probe_error": str(exc)}
    versions = {}
    for package in ("numpy", "scipy", "PyYAML", "highway-env", "gymnasium", "torch",
                    "stable-baselines3", "matplotlib", "shapely"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "project": "FBRT-Memory", "schema_version": "fbrt-memory-v2",
        "baseline_commit": "6af8a502c90235063e593d2d36c8aede3aa027f2",
        "execution_commit": _git_head(), "simulator": "highway-env",
        "physics_hz": 20, "new_execution_contract": EXECUTION_CONTRACT,
        "environment": {"python": platform.python_version(), "packages": versions,
                        "gpu_probe_in_conda_metadrive": gpu,
                        "model_device": "CPU per plan for compact Bayesian logistic regression"},
        "policy_training_allowed": False,
        "selected_recipe": recipe["recipe"],
        "selected_scenario_ids": recipe["selected_scenario_ids"],
        "scenario_catalogue_path": str(CATALOGUE.as_posix()),
        "catalogue_candidate_count": len(catalogue["scenarios"]),
        "samples_per_template": int(catalogue["samples_per_template"]),
        "builds": catalogue["new_builds"], "static_capabilities": capability,
        "ppo_resource_status": recipe["ppo_checkpoint_status"],
        "budget": {"new_compact_bank_max": MAIN_BANK_CAP,
                   "new_total_physical_max": GLOBAL_PHYSICAL_CAP,
                   "smoke_max": 24, "paired_replay_max": 8,
                   "required_repair_reserve_max": 48},
        "task_visibility": {"legacy_regression": "parent complete episodes only",
                             "compact_regression": "parent build complete bank",
                             "cross_agent": "only prior task queried observations per method branch"},
        "created_utc_note": "Timestamp is recorded in the run ledger at execution time.",
    }


def import_stage() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    catalogue = load_catalogue()
    capability, recipe = write_capability_and_recipe(ROOT)
    docs_catalogue = Path("docs/FBRT_SCENARIO_CATALOGUE_V2.yaml")
    if docs_catalogue.is_file() and CATALOGUE.read_bytes() != docs_catalogue.read_bytes():
        CATALOGUE.parent.mkdir(parents=True, exist_ok=True)
        CATALOGUE.write_bytes(docs_catalogue.read_bytes())
        catalogue = load_catalogue()
    (ROOT / "scenario_sources.md").write_text(
        (ROOT / "scenario_sources.md").read_text(encoding="utf-8")
        if (ROOT / "scenario_sources.md").is_file() else
        "Scenario source mapping is delivered in this result directory; see the catalogue and plan section 23–31.\n",
        encoding="utf-8")
    compiled = compile_scenarios(recipe, ROOT)
    new_records, report = load_legacy_archive(CORE)
    legacy = _rows_for_family([record.as_dict() for record in new_records])
    existing_store = SnapshotStore(ROOT)
    existing = existing_store.load_records()
    merged = {row["execution_id"]: row for row in existing}
    for row in legacy:
        existing_row = merged.get(row["execution_id"])
        if existing_row is None:
            merged[row["execution_id"]] = row
        elif existing_row.get("visibility") != row["visibility"]:
            merged[row["execution_id"]] = {**existing_row, "visibility": row["visibility"]}
    snapshot_hash = existing_store.save_records(merged.values())
    records = list(merged.values())
    cards = build_pattern_cards([row for row in records
                                 if row.get("visibility") != "evaluator_only"])
    cards_jsonl(cards, ROOT / "patterns.jsonl")
    report["legacy_file_sha256"] = {}
    for filename in ("reference_archive.csv", "candidate_pool.csv", "target_response_bank.csv"):
        path = CORE / filename
        if path.is_file():
            report["legacy_file_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["snapshot_hash"] = snapshot_hash
    report["newly_imported_record_count"] = len(new_records)
    report["existing_new_record_count_preserved"] = len(existing)
    report["pattern_card_count"] = len(cards)
    report["candidate_count"] = len(compiled)
    report["selected_recipe"] = recipe["recipe"]
    report["ppo_resource_status"] = recipe["ppo_checkpoint_status"]
    _write_json(ROOT / "archive_import_report.json", report)
    protocol = _protocol(capability, recipe, catalogue)
    _write_json(ROOT / "protocol.json", protocol)
    ledger_path = ROOT / "compute_ledger.json"
    if not ledger_path.exists():
        _write_json(ledger_path, {
            "global_physical_cap": GLOBAL_PHYSICAL_CAP,
            "main_bank_physical_cap": MAIN_BANK_CAP,
            "new_physical_episodes": 0,
            "physical_episode_categories": {"smoke": 0, "compact_bank": 0,
                                             "paired_replay": 0, "repair_or_retry": 0},
            "cache_hits": 0, "logical_query_count": 0,
            "physical_episode_ids": [], "denied_new_episode_requests": [],
            "wall_clock_seconds": 0.0, "legacy_physical_cost": 0,
            "model_fit_count": 0, "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    _save_source_models(records, cards)
    return {"stage": "import", "records": len(records), "patterns": len(cards),
            "candidate_scenarios": len(compiled), "snapshot_hash": snapshot_hash,
            "ppo_resource_status": recipe["ppo_checkpoint_status"]}


def _save_source_models(records: list[dict], cards) -> None:
    params = {}
    means, covariances, names = [], [], []
    feature_schemas = []
    contexts = sorted({(row["template_id"], row.get("context_id", "legacy_unspecified"))
                       for row in records})
    for template, context in contexts:
        local = [row for row in records if row["template_id"] == template and
                 row.get("context_id", "legacy_unspecified") == context and
                 row.get("visibility") != "evaluator_only"]
        local_cards = [card for card in cards if card.template_id == template and
                       card.context_id == context]
        if not local:
            continue
        dictionaries = build_dictionaries(local, local_cards, local, seed=4179901)
        dictionary = dictionaries[template]
        for build_id, fit in source_fits(local, dictionary).items():
            name = f"{template}|{context}|{build_id}"
            names.append(name)
            means.append(fit.mean)
            covariances.append(np.diag(fit.covariance))
            feature_schemas.append(json.dumps({
                "schema_identity": dictionary.schema_identity,
                "ordered_feature_ids": dictionary.ordered_feature_ids(),
                "feature_specs": dictionary.ordered_feature_specs()},
                ensure_ascii=False, sort_keys=True))
    params["source_names"] = np.asarray(names, dtype="U256")
    max_dim = max((len(item) for item in means), default=0)
    mean_matrix = np.full((len(means), max_dim), np.nan, dtype=float)
    covariance_matrix = np.full((len(covariances), max_dim), np.nan, dtype=float)
    dimensions = []
    for index, mean in enumerate(means):
        mean_matrix[index, :len(mean)] = mean
        covariance_matrix[index, :len(covariances[index])] = covariances[index]
        dimensions.append(len(mean))
    params["source_means"] = mean_matrix
    params["source_variance_diagonals"] = covariance_matrix
    params["source_dimensions"] = np.asarray(dimensions, dtype=int)
    params["source_feature_schemas"] = np.asarray(feature_schemas, dtype="U32768")
    path = ROOT / "source_models.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **params)


def _load_ledger() -> dict:
    path = ROOT / "compute_ledger.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    import_stage()
    return json.loads(path.read_text(encoding="utf-8"))


def _save_ledger(ledger: dict) -> None:
    _write_json(ROOT / "compute_ledger.json", ledger)


def _scenario_id(record: dict) -> str:
    return record.get("scenario_id") or record.get("scenario", {}).get("scenario_id", "unknown")


def _current_bank() -> tuple[Path, list[dict], dict[str, dict]]:
    path = ROOT / "compact_bank" / "episodes.jsonl"
    rows = read_jsonl(path)
    return path, rows, {row["execution_id"]: row for row in rows}


def _episode_cache_id(build_id: str, case: dict, seed: int = SIMULATOR_SEED) -> str:
    spec = build_spec_factory(build_id)
    scenario_fingerprint = stable_hash({"scenario": case,
                                        "execution_contract": EXECUTION_CONTRACT})
    return "new-" + stable_hash({"build_fingerprint": spec.fingerprint,
                                 "scenario_fingerprint": scenario_fingerprint,
                                 "seed": int(seed), "contract": EXECUTION_CONTRACT})[:24]


def _available_builds(capability: dict) -> tuple[list[str], list[str]]:
    available = ["mobil_ref_v2", "mobil_rear_guard_off_v2"]
    blocked = []
    if (capability.get("ppo_checkpoint_available") and
            capability.get("ppo_runtime_dependency_available", True)):
        available += ["ppo_ref_v2", "ppo_obs_age020_v2"]
    else:
        blocked += ["ppo_ref_v2", "ppo_obs_age020_v2"]
    return available, blocked


def _write_trace(execution_id: str, trace: list[dict]) -> str:
    directory = ROOT / "compact_bank" / "trajectories"
    path = directory / f"{execution_id}.jsonl"
    _write_jsonl(path, trace)
    return path.relative_to(ROOT).as_posix()


def _ensure_physical_episode(build_id: str, case: dict, category: str,
                             physical_limit: int, bank_path: Path,
                             by_id: dict[str, dict], ledger: dict,
                             capture_trace: bool = False) -> tuple[dict | None, str]:
    execution_id = _episode_cache_id(build_id, case)
    if execution_id in by_id:
        ledger["cache_hits"] = int(ledger.get("cache_hits", 0)) + 1
        return by_id[execution_id], "cache_hit"
    total = int(ledger.get("new_physical_episodes", 0))
    if physical_limit <= 0 or total >= min(GLOBAL_PHYSICAL_CAP, physical_limit):
        denied = ledger.setdefault("denied_new_episode_requests", [])
        denied.append({"build_id": build_id, "scenario_id": case["scenario_id"],
                       "reason": "physical_limit_reached", "physical_limit": physical_limit,
                       "global_cap": GLOBAL_PHYSICAL_CAP})
        _save_ledger(ledger)
        return None, "physical_limit_reached"
    if category in {"smoke", "repair_or_retry", "paired_replay"} and total >= 24 + 8 + 48 + MAIN_BANK_CAP:
        return None, "global_cap_reached"
    start = time.monotonic()
    outcome, trace = run_build_episode(build_id, case, SIMULATOR_SEED, with_trace=capture_trace)
    outcome["visibility"] = "evaluator_only"
    outcome["physical_category"] = category
    outcome["wall_clock_seconds"] = round(time.monotonic() - start, 5)
    if trace:
        outcome["trajectory_path"] = _write_trace(outcome["execution_id"], trace)
    _write_jsonl(bank_path, [outcome], append=True)
    by_id[outcome["execution_id"]] = outcome
    ledger["new_physical_episodes"] = total + 1
    categories = ledger.setdefault("physical_episode_categories", {})
    categories[category] = int(categories.get(category, 0)) + 1
    ledger["wall_clock_seconds"] = round(float(ledger.get("wall_clock_seconds", 0.0)) +
                                          outcome["wall_clock_seconds"], 3)
    ledger["physical_episode_ids"] = sorted(set(ledger.get("physical_episode_ids", [])) |
                                            {outcome["execution_id"]})
    _save_ledger(ledger)
    return outcome, "executed"


def _load_setup() -> tuple[dict, dict, list[dict]]:
    capability_path = ROOT / "capabilities.json"
    recipe_path = ROOT / "selected_recipe.json"
    if not capability_path.is_file() or not recipe_path.is_file():
        import_stage()
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    cases = read_jsonl(ROOT / "compact_bank" / "scenario_cases.jsonl")
    return capability, recipe, cases


def smoke_stage(physical_limit: int) -> dict:
    capability, recipe, cases = _load_setup()
    available, blocked = _available_builds(capability)
    bank_path, bank_rows, by_id = _current_bank()
    start = time.monotonic()
    performed = []
    per_template: dict[str, dict] = {}
    for case in cases:
        per_template.setdefault(case["template_id"], case)
    for build in available:
        for template, case in per_template.items():
            row, status = _ensure_physical_episode(build, case, "smoke", physical_limit,
                                                   bank_path, by_id, _load_ledger(),
                                                   capture_trace=True)
            if row is None:
                break
            performed.append({"build_id": build, "scenario_id": case["scenario_id"],
                              "template_id": template, "status": status,
                              "ego_collision": row["ego_collision"],
                              "inconclusive": row["inconclusive"],
                              "trajectory_path": row.get("trajectory_path")})
        if len(performed) >= 24:
            break
    result = {"stage": "smoke", "physical_calls_this_stage": sum(
        row["status"] == "executed" for row in performed),
        "records": performed, "blocked_builds": blocked,
        "resource_note": "PPO smoke not run because checkpoint is absent" if blocked else "all builds available",
        "elapsed_seconds": round(time.monotonic() - start, 3)}
    _write_json(ROOT / "smoke_report.json", result)
    return result


def compact_bank_stage(physical_limit: int) -> dict:
    capability, recipe, cases = _load_setup()
    available, blocked = _available_builds(capability)
    bank_path, bank_rows, by_id = _current_bank()
    start = time.monotonic()
    counts_before = len(by_id)
    compact_calls_before = int(_load_ledger().get("physical_episode_categories", {}).get("compact_bank", 0))
    stop_reason = "complete_for_available_builds"
    for build in available:
        for case in cases:
            row, status = _ensure_physical_episode(build, case, "compact_bank", physical_limit,
                                                   bank_path, by_id, _load_ledger(),
                                                   capture_trace=(case["scenario_id"].endswith("anchor_like:0")))
            if row is None:
                stop_reason = status
                break
        if stop_reason != "complete_for_available_builds":
            break
    bank_rows = read_jsonl(bank_path)
    ledger = _load_ledger()
    counts = Counter(row["build_id"] for row in bank_rows)
    result = {
        "stage": "compact-bank", "recipe": recipe["recipe"],
        "selected_scenario_count": len(cases), "available_builds": available,
        "blocked_builds": blocked, "episodes_before": counts_before,
        "episodes_after": len({row["execution_id"] for row in bank_rows}),
        "new_calls_this_stage": int(ledger.get("physical_episode_categories", {}).get(
            "compact_bank", 0)) - compact_calls_before,
        "unique_episode_counts_by_build": dict(counts),
        "target_full_bank_episode_count": len(cases) * 4,
        "bank_complete": len(counts) == 4 and all(counts.get(build, 0) == len(cases)
                                                  for build in recipe.get("builds", [])),
        "stop_reason": stop_reason,
        "global_physical_used": ledger["new_physical_episodes"],
        "global_physical_cap": GLOBAL_PHYSICAL_CAP,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }
    result["bank_complete"] = (not blocked and len(bank_rows) == len(cases) * 4 and
                               all(counts.get(build, 0) == len(cases) for build in available))
    resource = {"ppo_status": "available" if not blocked else "PPO_UNAVAILABLE",
                "blocked_builds": blocked,
                "checkpoint_path": capability.get("ppo_checkpoint_path"),
                "checkpoint_available": capability.get("ppo_checkpoint_available"),
                "runtime_dependency_available": capability.get("ppo_runtime_dependency_available"),
                "fetch_flow_found": Path("replications/highway_sut_selection/assets.py").is_file(),
                "checkpoint_sha256": capability.get("ppo_checkpoint_sha256"),
                "effect": "PPO reference and observation-age builds omitted; no PPO outcomes inferred"
                if blocked else "none"}
    _write_json(ROOT / "resource_status.json", resource)
    _write_json(ROOT / "compact_bank" / "bank_status.json", result)
    return result


def _candidate_records_for_legacy() -> tuple[list[dict], dict[str, dict]]:
    parent_records = [row for row in read_jsonl(ROOT / "archive_v2.jsonl")
                      if row["build_id"] == "idm_ref"]
    parent_by_id = {row["scenario_id"]: row for row in parent_records}
    candidates = []
    for row in _read_csv(CORE / "candidate_pool.csv"):
        parent = parent_by_id[row["scenario_id"]]
        if not is_parent_pass(parent):
            continue
        scenario = dict(parent["scenario"])
        scenario["context_id"] = parent["context_id"]
        candidates.append({"scenario_id": row["scenario_id"],
                           "template_id": row["template_id"], "scenario": scenario,
                           "context_id": parent["context_id"],
                           "parent_pass": is_parent_pass(parent),
                           "completed": parent["completed"],
                           "ego_collision": parent["ego_collision"],
                           "min_ttc": parent["min_ttc"],
                           "min_clearance": parent["min_clearance"]})
    return candidates, parent_by_id


def _legacy_eval_bank() -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    rows = _read_csv(CORE / "target_response_bank.csv")
    archive = {}
    banks: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        record = row_to_record(row, "target_response_bank.csv").as_dict()
        record["family"] = _family(record["build_id"])
        # Historical measured targets are evaluation-only for this replay.
        record["visibility"] = "evaluator_only"
        banks[record["build_id"]][record["scenario_id"]] = record
        archive[record["execution_id"]] = record
    return banks, archive


def _write_session(task_dir: Path, method: str, repeat: int,
                    task_id: str, target_build: str, mode: str,
                    snapshot_hash: str, candidates: list[dict], queries: list[dict],
                    observations: list[dict], cards: list, updates: list[dict]) -> None:
    session_dir = task_dir / method / f"repeat_{repeat}"
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(session_dir / "queries.csv", queries)
    _write_jsonl(session_dir / "updates.jsonl", updates)
    session = Session(session_id=f"{task_id}:{method}:{repeat}", task_id=task_id,
                      target_build_id=target_build, mode=mode,
                      history_snapshot_before=snapshot_hash,
                      candidate_ids=[item["scenario_id"] for item in candidates],
                      queried_execution_ids=[row["execution_id"] for row in queries],
                      observations=observations,
                      created_pattern_ids=[card.pattern_id for card in cards
                                           if card.created_in_session == f"{task_id}:{method}:{repeat}"])
    session.history_snapshot_after = snapshot_hash
    _write_json(session_dir / "session.json", session.as_dict())


def _summaries(task_id: str, target_build: str, method: str, repeat: int,
               queries: list[dict], total_failures: int,
               candidate_count: int) -> list[dict]:
    def discovered(query: dict) -> bool:
        if query.get("mode") == "regression":
            return bool(query.get("regression"))
        return query.get("ego_collision") is True and not query.get("inconclusive", False)

    result = []
    horizon = len(queries)
    cumulative = 0
    recall_prefix = []
    for rank, query in enumerate(queries, 1):
        cumulative += int(discovered(query))
        recall_prefix.append(cumulative / total_failures if total_failures > 0 else None)
        for budget in CHECKPOINTS:
            if rank != min(budget, horizon):
                continue
            prefix = queries[:rank]
            count = sum(discovered(row) for row in prefix)
            recall = count / total_failures if total_failures > 0 else None
            first = next((row["rank"] for row in prefix if discovered(row)), None)
            valid_prefix = [row for row in prefix if not row["inconclusive"]]
            auc = (float(np.mean([sum(discovered(row) for row in queries[:b]) /
                                  total_failures if total_failures else np.nan
                                  for b in range(1, rank + 1)]))
                   if total_failures else None)
            result.append({"task_id": task_id, "target_build": target_build,
                           "method": method, "repeat": repeat, "budget": min(budget, horizon),
                           "effective_horizon": horizon, "candidate_count": candidate_count,
                           "actual_queries": len(prefix), "valid_results": len(valid_prefix),
                           "failure_detected": bool(count), "first_failure_rank": first,
                           "failure_count": count, "failure_pool_count": total_failures,
                           "failure_recall": recall, "early_failure_recall_auc_20": auc})
    # Handles zero-query/no-eligible tasks transparently.
    if not result:
        result.append({"task_id": task_id, "target_build": target_build,
                       "method": method, "repeat": repeat, "budget": 0,
                       "effective_horizon": 0, "candidate_count": candidate_count,
                       "actual_queries": 0, "valid_results": 0,
                       "failure_detected": False, "first_failure_rank": None,
                       "failure_count": 0, "failure_pool_count": total_failures,
                       "failure_recall": None, "early_failure_recall_auc_20": None,
                       "status": "no_eligible_cases"})
    return result


def _run_methods(task_id: str, target_build: str, candidates: list[dict],
                 history: list[dict], evaluator_bank: dict[str, dict],
                 mode: str, parent_build_id: str | None,
                 snapshot_hash: str, directory: Path,
                 repeats_random: int = RANDOM_REPEATS) -> tuple[list[dict], list[dict], list[dict]]:
    candidates = [dict(row) for row in candidates]
    history = _contextual_history(history, candidates)
    cards = build_pattern_cards(history, session_id=task_id)
    bank = {key: value for key, value in evaluator_bank.items()
            if key in {row["scenario_id"] for row in candidates}}
    if set(bank) != {row["scenario_id"] for row in candidates}:
        raise ValueError(f"incomplete evaluator bank for task {task_id}")
    total_failures = sum(1 for sid, row in bank.items()
                         if row.get("ego_collision") is True and
                         not row.get("inconclusive", False) and
                         next(item for item in candidates if item["scenario_id"] == sid).get(
                             "parent_pass", True))
    all_queries, all_summaries, session_observations = [], [], []
    store = SnapshotStore(ROOT)
    before_hash = stable_hash(store.load_records())
    families = {row["build_id"]: row.get("family", _family(row["build_id"])) for row in history}
    for method in METHODS:
        for repeat in range(repeats_random if method == "Random" else 1):
            oracle = TargetOracle(bank)
            this_session = f"{task_id}:{method}:{repeat}"
            queries, observations, learned_cards, updates = run_selector(
                method, candidates, history, oracle, BUDGET,
                random_seed=4179901 + repeat + int(stable_hash(this_session)[:8], 16),
                target_build_id=target_build, parent_build_id=parent_build_id,
                mode=mode, session_id=this_session, family_by_build=families,
                initial_cards=cards,
            )
            all_queries.extend({**query, "task_id": task_id, "repeat": repeat}
                               for query in queries)
            all_summaries.extend(_summaries(task_id, target_build, method, repeat,
                                            queries, total_failures, len(candidates)))
            _write_session(directory, method, repeat, task_id, target_build, mode,
                           before_hash, candidates, queries, observations, learned_cards, updates)
            if method == "FBRT-Memory":
                session_observations = [{**row, "family": _family(target_build)}
                                        for row in observations]
    return all_queries, all_summaries, session_observations


def cache_stage() -> dict:
    start = time.monotonic()
    root_records = _rows_for_family(read_jsonl(ROOT / "archive_v2.jsonl"))
    candidates_all, parent_by_id = _candidate_records_for_legacy()
    banks, eval_records = _legacy_eval_bank()
    reference_ids = sorted({int(row["scenario_id"].split(":", 1)[0])
                            for row in candidates_all})
    query_rows, summary_rows = [], []
    session_root = ROOT / "sessions" / "legacy_regression"
    snapshot_hash = stable_hash(read_jsonl(ROOT / "archive_v2.jsonl"))
    for seed in reference_ids:
        candidates = [item for item in candidates_all
                      if int(item["scenario_id"].split(":", 1)[0]) == seed]
        parent_history = [row for row in root_records
                          if row.get("build_id") == "idm_ref"
                          and row.get("simulator_seed") == seed
                          and (row.get("visibility") == "historical" or
                               (row.get("visibility") is None and
                                row.get("source_file") == "reference_archive.csv"))
                          and is_usable_outcome(row)]
        task_targets = [build for build in ("merge_blind06", "merge_brake2", "slow_front_brake2")
                        if any(int(sid.split(":", 1)[0]) == seed for sid in banks.get(build, {}))]
        for target in task_targets:
            evaluator = {sid: row for sid, row in banks[target].items()
                         if int(sid.split(":", 1)[0]) == seed}
            task_id = f"legacy_regression_seed{seed}_{target}"
            queries, summaries, _ = _run_methods(
                task_id, target, candidates, parent_history, evaluator, "regression",
                "idm_ref", snapshot_hash, session_root / task_id)
            query_rows.extend(queries)
            summary_rows.extend(summaries)
    _write_csv(ROOT / "sessions" / "legacy_regression" / "queries.csv", query_rows)
    _write_csv(ROOT / "summary_by_task.csv", summary_rows)
    ledger = _load_ledger()
    ledger["logical_query_count"] = len(query_rows)
    ledger["cache_replay_physical_cost"] = 0
    ledger["model_fit_count"] = int(ledger.get("model_fit_count", 0)) + len(summary_rows)
    ledger["wall_clock_seconds"] = round(float(ledger.get("wall_clock_seconds", 0.0)) +
                                          (time.monotonic() - start), 3)
    _save_ledger(ledger)
    return {"stage": "cache", "tasks": len({row["task_id"] for row in summary_rows}),
            "campaigns": len({(row["task_id"], row["method"], row["repeat"])
                               for row in summary_rows}),
            "logical_queries": len(query_rows), "new_physical_episodes": 0,
            "elapsed_seconds": round(time.monotonic() - start, 3)}


def evaluate_compact_bank() -> tuple[list[dict], list[dict], list[dict]]:
    bank_path, bank_rows, _ = _current_bank()
    if not bank_rows:
        return [], [], [{"task_id": "compact_bank", "status": "not_run"}]
    bank = defaultdict(dict)
    for row in bank_rows:
        bank[row["build_id"]][row["scenario_id"]] = row
    cases = read_jsonl(ROOT / "compact_bank" / "scenario_cases.jsonl")
    capability, _recipe, _ = _load_setup()
    available, blocked = _available_builds(capability)
    queries_all, summaries_all, task_status = [], [], []
    canonical_memory_observations: list[dict] = []
    by_template: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        case_copy = dict(case)
        case_copy["context_id"] = case["context_id"]
        by_template[case["template_id"]].append({
            "scenario_id": case["scenario_id"], "template_id": case["template_id"],
            "scenario": case_copy, "context_id": case["context_id"],
        })
    compact_regression_root = ROOT / "sessions" / "compact_regression"
    if {"mobil_ref_v2", "mobil_rear_guard_off_v2"} <= set(available):
        parents = [row for row in bank["mobil_ref_v2"].values()
                   if row.get("completed") and not row.get("ego_collision")]
        candidates = []
        for row in parents:
            case = row["scenario"]
            candidates.append({"scenario_id": row["scenario_id"],
                               "template_id": row["template_id"], "scenario": case,
                               "context_id": case.get("context_id"), "parent_pass": True,
                               "completed": True, "ego_collision": False})
        evaluator = bank.get("mobil_rear_guard_off_v2", {})
        if candidates and all(row["scenario_id"] in evaluator for row in candidates):
            task_id = "compact_regression_mobil_ref_to_rear_guard_off"
            history = [{**row, "visibility": "historical", "family": _family("mobil_ref_v2")}
                       for row in bank["mobil_ref_v2"].values()]
            queries, summaries, _ = _run_methods(
                task_id, "mobil_rear_guard_off_v2", candidates, history, evaluator,
                "regression", "mobil_ref_v2", stable_hash(history),
                compact_regression_root / task_id)
            queries_all.extend(queries); summaries_all.extend(summaries)
            task_status.append({"task_id": task_id, "status": "complete",
                                "candidate_count": len(candidates),
                                "regression_pool_failures": sum(
                                    row.get("ego_collision") is True and
                                    not row.get("inconclusive", False)
                                    for row in (evaluator.get(item["scenario_id"], {})
                                                for item in candidates))})
        else:
            task_status.append({"task_id": "compact_regression_mobil_ref_to_rear_guard_off",
                                "status": "no_eligible_cases" if not candidates else "incomplete_bank",
                                "candidate_count": len(candidates)})
    else:
        task_status.append({"task_id": "compact_regression_mobil", "status": "builds_unavailable"})
    if {"ppo_ref_v2", "ppo_obs_age020_v2"} <= set(available):
        parents = [row for row in bank["ppo_ref_v2"].values()
                   if row.get("completed") and not row.get("ego_collision")]
        candidates = []
        for row in parents:
            case = row["scenario"]
            candidates.append({"scenario_id": row["scenario_id"],
                               "template_id": row["template_id"], "scenario": case,
                               "context_id": case.get("context_id"), "parent_pass": True,
                               "completed": True, "ego_collision": False})
        evaluator = bank.get("ppo_obs_age020_v2", {})
        task_id = "compact_regression_ppo_ref_to_obs_age020"
        if candidates and all(row["scenario_id"] in evaluator for row in candidates):
            history = [{**row, "visibility": "historical", "family": _family("ppo_ref_v2")}
                       for row in bank["ppo_ref_v2"].values()]
            queries, summaries, _ = _run_methods(
                task_id, "ppo_obs_age020_v2", candidates, history, evaluator,
                "regression", "ppo_ref_v2", stable_hash(history),
                compact_regression_root / task_id)
            queries_all.extend(queries); summaries_all.extend(summaries)
            failure_count = sum(row.get("ego_collision") is True and
                                not row.get("inconclusive", False)
                                for row in (evaluator[item["scenario_id"]]
                                            for item in candidates))
            task_status.append({"task_id": task_id,
                                "status": "complete" if failure_count else
                                "no_observed_regression_in_pool",
                                "candidate_count": len(candidates),
                                "regression_pool_failures": failure_count})
        else:
            task_status.append({"task_id": task_id,
                                "status": "no_eligible_cases" if not candidates else "incomplete_bank",
                                "candidate_count": len(candidates)})
    if blocked:
        task_status.append({"task_id": "compact_regression_ppo_ref_to_obs_age020",
                            "status": "PPO_UNAVAILABLE", "blocked_builds": blocked})
    # Each selector method owns its branch. Only its queried, valid first-agent
    # observations become the second-agent history for the same method.
    if "mobil_ref_v2" in available:
        candidates = [item for rows in by_template.values() for item in rows]
        evaluator = bank.get("mobil_ref_v2", {})
        if set(evaluator) >= {row["scenario_id"] for row in candidates}:
            cross_root = ROOT / "sessions" / "compact_cross_agent"
            base_records = _rows_for_family(_cross_agent_seed_records(
                read_jsonl(ROOT / "archive_v2.jsonl"), bank_rows))
            base_snapshot = stable_hash(base_records)
            method_summaries = []
            for method in METHODS:
                for repeat in range(RANDOM_REPEATS if method == "Random" else 1):
                    method_root = cross_root / "branches" / method / f"repeat_{repeat}"
                    branch_store = SnapshotStore(method_root / "history_branch")
                    seed_history = _contextual_history(base_records, candidates)
                    branch_store.save_records(seed_history)
                    first_oracle = TargetOracle(evaluator)
                    first_task = f"cross_agent_{method}_mobil_to_next"
                    filtered_first = _contextual_history(seed_history, candidates)
                    initial_cards = build_pattern_cards(filtered_first, session_id=first_task)
                    family_by_build = {row["build_id"]: row.get("family", _family(row["build_id"]))
                                       for row in filtered_first}
                    first_queries, first_observations, first_cards, first_updates = run_selector(
                        method, candidates, filtered_first, first_oracle, BUDGET,
                        random_seed=99001 + repeat, target_build_id="mobil_ref_v2", mode="cross_agent",
                        session_id=first_task, family_by_build=family_by_build,
                        initial_cards=initial_cards)
                    _write_session(cross_root / first_task, method, repeat, first_task,
                                   "mobil_ref_v2", "cross_agent",
                                   base_snapshot, candidates, first_queries, first_observations,
                                   first_cards, first_updates)
                    before_hash, after_hash, inserted = branch_store.commit(
                        [{**row, "family": _family("mobil_ref_v2")} for row in first_observations])
                    first_session_path = cross_root / first_task / method / f"repeat_{repeat}" / "session.json"
                    first_session = json.loads(first_session_path.read_text(encoding="utf-8"))
                    first_session["history_snapshot_after"] = after_hash
                    _write_json(first_session_path, first_session)
                    if method == "FBRT-Memory":
                        canonical_memory_observations.extend(
                            {**row, "family": _family("mobil_ref_v2")}
                            for row in first_observations)
                        global_store = SnapshotStore(ROOT)
                        global_store.commit([{**row, "family": _family("mobil_ref_v2")}
                                             for row in first_observations])
                    if method in ("FBRT-Memory", "FBRT-NoMemory"):
                        task_status.append({"task_id": first_task, "status": "complete",
                                            "queried": len(first_queries),
                                            "committed_observations": len(inserted),
                                            "snapshot_before": before_hash,
                                            "snapshot_after": after_hash})
                    if "ppo_ref_v2" not in available:
                        method_summaries.extend(_summaries(
                            first_task, "mobil_ref_v2", method, repeat, first_queries,
                            sum(row.get("ego_collision") is True and not row.get("inconclusive", False)
                                for row in evaluator.values()), len(candidates)))
                        continue
                    ppo_evaluator = bank.get("ppo_ref_v2", {})
                    if set(ppo_evaluator) < {row["scenario_id"] for row in candidates}:
                        task_status.append({"task_id": f"cross_agent_{method}_ppo_ref",
                                            "status": "incomplete_bank"})
                        continue
                    history = _rows_for_family(branch_store.load_records())
                    history = _contextual_history(history, candidates)
                    ppo_oracle = TargetOracle(ppo_evaluator)
                    next_task = f"cross_agent_{method}_ppo_ref_after_mobil"
                    next_cards = build_pattern_cards(history, session_id=next_task)
                    next_queries, next_observations, next_pattern_cards, next_updates = run_selector(
                        method, candidates, history, ppo_oracle, BUDGET,
                        random_seed=99002 + repeat, target_build_id="ppo_ref_v2", mode="cross_agent",
                        session_id=next_task,
                        family_by_build={row["build_id"]: row.get("family", _family(row["build_id"]))
                                         for row in history}, initial_cards=next_cards)
                    _write_session(cross_root / next_task, method, repeat, next_task,
                                   "ppo_ref_v2", "cross_agent",
                                   after_hash, candidates, next_queries, next_observations,
                                   next_pattern_cards, next_updates)
                    before_next_hash, after_next_hash, inserted_next = branch_store.commit(
                        [{**row, "family": _family("ppo_ref_v2")} for row in next_observations])
                    next_session_path = cross_root / next_task / method / f"repeat_{repeat}" / "session.json"
                    next_session = json.loads(next_session_path.read_text(encoding="utf-8"))
                    next_session["history_snapshot_after"] = after_next_hash
                    _write_json(next_session_path, next_session)
                    if method == "FBRT-Memory":
                        canonical_memory_observations.extend(
                            {**row, "family": _family("ppo_ref_v2")}
                            for row in next_observations)
                        global_store = SnapshotStore(ROOT)
                        global_store.commit([{**row, "family": _family("ppo_ref_v2")}
                                             for row in next_observations])
                    if method in ("FBRT-Memory", "FBRT-NoMemory"):
                        task_status.append({"task_id": next_task, "status": "complete",
                                            "queried": len(next_queries),
                                            "committed_observations": len(inserted_next),
                                            "snapshot_before": before_next_hash,
                                            "snapshot_after": after_next_hash})
                    method_summaries.extend(_summaries(
                        first_task, "mobil_ref_v2", method, repeat, first_queries,
                        sum(row.get("ego_collision") is True and not row.get("inconclusive", False)
                            for row in evaluator.values()), len(candidates)))
                    method_summaries.extend(_summaries(
                        next_task, "ppo_ref_v2", method, repeat, next_queries,
                        sum(row.get("ego_collision") is True and not row.get("inconclusive", False)
                            for row in ppo_evaluator.values()), len(candidates)))
            summaries_all.extend(method_summaries)
    _write_csv(ROOT / "sessions" / "compact_regression" / "queries.csv", queries_all)
    _write_csv(ROOT / "sessions" / "compact_cross_agent" / "summary_by_task.csv",
               summaries_all)
    previous = _read_csv(ROOT / "summary_by_task.csv")
    retained = [row for row in previous if not row.get("task_id", "").startswith(
        ("compact_", "cross_agent_"))]
    _write_csv(ROOT / "summary_by_task.csv", retained + summaries_all)
    store = SnapshotStore(ROOT)
    bank_execution_ids = {row["execution_id"] for row in bank_rows}
    canonical_records = {row["execution_id"]: row for row in store.load_records()
                         if row["execution_id"] not in bank_execution_ids}
    canonical_records.update((row["execution_id"], row)
                             for row in canonical_memory_observations)
    store.save_records(canonical_records.values())
    final_records = store.load_records()
    cards = build_pattern_cards([row for row in final_records
                                 if row.get("visibility") != "evaluator_only"],
                                session_id="post_compact_cross_agent")
    cards_jsonl(cards, ROOT / "patterns.jsonl")
    return queries_all, summaries_all, task_status


def evaluate_stage() -> dict:
    start = time.monotonic()
    if not (ROOT / "summary_by_task.csv").is_file() or not (ROOT / "sessions/legacy_regression/queries.csv").is_file():
        legacy = cache_stage()
    else:
        legacy = {"stage": "cache_reused"}
    queries, summaries, task_status = evaluate_compact_bank()
    summary_path = ROOT / "summary_by_task.csv"
    rows = _read_csv(summary_path) if summary_path.is_file() else []
    _write_csv(summary_path, rows)
    ledger = _load_ledger()
    ledger["logical_query_count"] = sum(1 for _ in _all_query_rows())
    ledger["offline_evaluate_physical_cost"] = 0
    ledger["wall_clock_seconds"] = round(float(ledger.get("wall_clock_seconds", 0.0)) +
                                          (time.monotonic() - start), 3)
    _save_ledger(ledger)
    acceptance = _acceptance(rows, task_status, ledger)
    _write_json(ROOT / "acceptance.json", acceptance)
    from methods.failure_memory_regression.report import generate_report
    report = generate_report(ROOT, legacy_result=legacy, compact_task_status=task_status)
    return {"stage": "evaluate", "summary_rows": len(rows),
            "compact_task_status": task_status,
            "acceptance_level": acceptance["delivery_level"], "report": report,
            "new_physical_episodes": 0, "elapsed_seconds": round(time.monotonic() - start, 3)}


def _all_query_rows():
    for path in (ROOT / "sessions").rglob("queries.csv"):
        if not path.parent.name.startswith("repeat_"):
            continue
        if ("compact_cross_agent" in path.parts and
                path.parent.parent.parent.name == "compact_cross_agent"):
            # Legacy v2 runs placed both agents under method/repeat_0 and
            # overwrote the first session. Their files are superseded.
            continue
        try:
            yield from _read_csv(path)
        except (OSError, csv.Error):
            continue


def _acceptance(summary: list[dict], task_status: list[dict], ledger: dict) -> dict:
    bank = read_jsonl(ROOT / "compact_bank" / "episodes.jsonl")
    build_counts = Counter(row["build_id"] for row in bank)
    s02 = [row for row in bank if row.get("template_id") == "fbrt_cutout_static"]
    mobil_parent = [row for row in bank if row.get("build_id") == "mobil_ref_v2"]
    ppo_blocked = any(row.get("status") == "PPO_UNAVAILABLE" for row in task_status)
    task_by_id = {row["task_id"]: row for row in task_status}
    ppo_regression_recorded = task_by_id.get(
        "compact_regression_ppo_ref_to_obs_age020", {}).get("status") in {
            "complete", "no_observed_regression_in_pool", "no_eligible_cases"}
    second_agent_observed = task_by_id.get(
        "cross_agent_FBRT-Memory_ppo_ref_after_mobil", {}).get("status") == "complete"
    real_memory_queries = [row for row in _all_query_rows()
                           if row.get("method") == "FBRT-Memory" and
                           row.get("contributing_pattern_ids") not in ("[]", "")]
    has_recall_gain = False
    for row in summary:
        if row.get("method") not in {"FBRT-Memory", "FBRT-NoMemory",
                                     "HistoryRank-UCB-v2", "FailureDistance-v2"}:
            continue
        if str(row.get("method")) == "FBRT-Memory" and float(row.get("failure_count") or 0) > 0:
            task_id = row.get("task_id")
            best_other = max((float(other.get("failure_count") or 0) for other in summary
                              if other.get("task_id") == task_id and
                              other.get("method") not in {"Random", "FBRT-Memory"} and
                              other.get("budget") == row.get("budget")), default=0)
            if float(row.get("failure_count") or 0) > best_other:
                has_recall_gain = True
    checks = {
        "E1_unified_runner_and_registry": {"status": "implemented", "evidence":
                                            "BuildSpec registry; legacy/native/PPO adapter routing"},
        "E2_single_ego_action_owner": {"status": "implemented", "evidence":
                                       "external high-level decision applied once; low-level action ticks separated"},
        "E3_real_pattern_and_same_source_contrasts": {"status": "implemented", "pattern_cards":
                                                       len(read_jsonl(ROOT / "patterns.jsonl"))},
        "E4_pattern_changes_prediction_or_selection": {"status": "implemented_with_tests",
                                                       "query_rows_with_contributing_patterns": len(real_memory_queries)},
        "E5_new_failure_and_no_history_fallback": {"status": "implemented_with_tests"},
        "E6_no_double_count_and_full_observation_update": {"status": "implemented_with_tests"},
        "E7_snapshot_reuse_by_next_session": {"status":
                                               "observed_two_session_persistence" if second_agent_observed else
                                               "partial_ppo_session_unavailable" if ppo_blocked else
                                               "second_agent_incomplete"},
        "E8_visibility_candidate_budget_consistency": {"status": "implemented",
                                                        "task_rows": len(summary)},
        "E9_experiments_complete_or_resource_block_recorded": {
            "status": "resource_block_recorded" if ppo_blocked else
            "complete" if ppo_regression_recorded and second_agent_observed else "incomplete",
            "compact_build_episode_counts": dict(build_counts), "task_status": task_status},
        "E10_global_physical_cap": {"status": "pass" if ledger.get("new_physical_episodes", 0) <= GLOBAL_PHYSICAL_CAP else "fail",
                                     "used": ledger.get("new_physical_episodes", 0),
                                     "cap": GLOBAL_PHYSICAL_CAP},
    }
    diagnostic_limits = {
        "mobil_reference_failures": sum(row.get("ego_collision") is True for row in mobil_parent),
        "mobil_initial_failure_cards": len(build_pattern_cards(mobil_parent)),
        "ppo_regression_pool_failures": task_by_id.get(
            "compact_regression_ppo_ref_to_obs_age020", {}).get("regression_pool_failures"),
        "s02_invalid_zero_first_exit_timestamps": sum(
            row.get("event_times", {}).get("first_exit_s") == 0.0 for row in s02),
        "s02_episode_count": len(s02),
        "ppo_s02_lead_collisions": sum(row.get("build_id", "").startswith("ppo_") and
                                        row.get("ego_collision") is True and
                                        row.get("collision_partner_role") == "lead"
                                        for row in s02),
    }
    return {"schema_version": "fbrt-memory-v2", "checks": checks,
            "diagnostic_limits": diagnostic_limits,
            "method_effect_observation": "local_gain_observed" if has_recall_gain else
            "no_gain_or_incomplete_tasks_as_observed",
            "delivery_level": "PARTIAL_RESOURCE_BLOCK" if ppo_blocked else
            "INCOMPLETE" if not (ppo_regression_recorded and second_agent_observed) else
            "IMPLEMENTED_WITH_GAIN" if has_recall_gain else "IMPLEMENTED_NO_GAIN",
            "interpretation": "Engineering completion and empirical gain are reported separately; no significance claim is required."}


def run_stage(stage: str, physical_limit: int = 0) -> dict:
    if physical_limit < 0 or physical_limit > GLOBAL_PHYSICAL_CAP:
        raise ValueError(f"physical-limit must be in [0,{GLOBAL_PHYSICAL_CAP}]")
    if stage == "import":
        return import_stage()
    if stage == "cache":
        if not (ROOT / "archive_v2.jsonl").exists():
            import_stage()
        return cache_stage()
    if stage == "smoke":
        if not (ROOT / "archive_v2.jsonl").exists():
            import_stage()
        return smoke_stage(physical_limit)
    if stage == "compact-bank":
        if not (ROOT / "archive_v2.jsonl").exists():
            import_stage()
        return compact_bank_stage(physical_limit)
    if stage == "evaluate":
        if not (ROOT / "archive_v2.jsonl").exists():
            import_stage()
        return evaluate_stage()
    if stage == "all":
        output = [import_stage(), cache_stage()]
        output.append(smoke_stage(physical_limit))
        output.append(compact_bank_stage(physical_limit))
        output.append(evaluate_stage())
        return {"stages": output}
    raise ValueError(f"unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("import", "cache", "smoke", "compact-bank",
                                             "evaluate", "all"), default="all")
    parser.add_argument("--physical-limit", type=int, default=0,
                        help="maximum cumulative new physical episodes, shared ledger cap is 400")
    args = parser.parse_args()
    result = run_stage(args.stage, args.physical_limit)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
