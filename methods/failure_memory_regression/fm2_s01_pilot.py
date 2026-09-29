"""Full-pool S01 failure discovery experiment with isolated target feedback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist

from highway_sim_env.envs.fbrt_unified_env import FBRTUnifiedEnv, run_build_episode
from methods.failure_memory_regression.fm2_memory import build_memory, memory_arrays
from methods.failure_memory_regression.fm2_model import FM2Model
from methods.failure_memory_regression.fm2_schema import CONFIG, SOURCES, TARGET, coordinates, valid_label
from methods.failure_memory_regression.fm2_selector import run_fm2
from methods.failure_memory_regression.fm2_train import train
from methods.failure_memory_regression.pattern_memory import active_values
from methods.failure_memory_regression.selector import TargetOracle, _history_rank, run_selector
from sut_algorithms.highway_env.registry import build_spec_factory


ROOT = Path("results/method_chains/failure_memory_regression/fm2_s01_full_history")
SOURCE_MANIFEST = Path("results/method_chains/failure_memory_regression/scenario_sampling_development/scenario_manifest.jsonl")
EXPERIMENT_CONFIG = Path(__file__).resolve().parent / "configs" / "fm2_s01_full_history.json"
MODULE_DIR = Path(__file__).resolve().parent
BASELINES = ("Random", "FailureDistance-v2", "HistoryRank-UCB-v2", "FBRT-Memory-Exploit-v3")
CHECKPOINTS = (5, 10, 20, 30, 50)


def _config() -> dict:
    config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    if (config["family"], config["target_build"], config["budget"]) != ("S01", TARGET, 50):
        raise ValueError("config violates the frozen S01 target and query budget")
    if (config["candidate_count"] != 2048 or
            config["history_candidate_count"] != config["candidate_count"] or
            tuple(config["sources"]) != SOURCES):
        raise ValueError("all historical builds must cover the same 2048 S01 scenarios")
    if len(config["training_seeds"]) != 1 or config["selector_repeats"] != 1:
        raise ValueError("this development run uses one training and selector seed")
    return config


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _manifest() -> list[dict]:
    rows = _jsonl(ROOT / "scenario_manifest.jsonl")
    if len(rows) != _config()["candidate_count"] or any(row["catalogue_id"] != "S01" or
                                row["sample_index"] != i for i, row in enumerate(rows)):
        raise ValueError("manifest does not match the frozen S01 Sobol pool")
    return rows


def _historical_bank(scenarios: list[dict]) -> list[dict]:
    rows = _jsonl(ROOT / "history/source_response_bank.jsonl")
    expected = {(build, scene["scenario_id"]) for build in SOURCES for scene in scenarios}
    actual = [(row["build_id"], row["scenario_id"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("every historical build must have one response for every S01 scenario")
    return rows


def _target_response_bank(scenarios: list[dict]) -> dict[str, dict]:
    rows = _jsonl(ROOT / "target/full_response_bank.jsonl")
    expected = {scene["scenario_id"] for scene in scenarios}
    actual = {row["scenario_id"]: row for row in rows}
    if len(rows) != len(expected) or set(actual) != expected or any(
            row.get("build_id") != TARGET for row in rows):
        raise ValueError("target truth bank must cover every frozen S01 scenario exactly once")
    return actual


def _environment() -> dict:
    return {"python": sys.executable, "python_version": platform.python_version(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_count": torch.cuda.device_count(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
            "platform": platform.platform()}


def _smoke(rows: list[dict]) -> list[dict]:
    from copy import deepcopy

    case = rows[0]
    records = []
    for axis, limits in case["research_bounds"].items():
        for name, value in (("low", limits[0]), ("high", limits[1])):
            changed = deepcopy(case)
            changed["active_parameters"][axis] = value
            env = FBRTUnifiedEnv(build_spec_factory("idm_ref"), changed)
            try:
                env.reset(seed=changed["simulator_seed"])
                lead = env.actors["lead"]
                state = {"lead_x": float(lead.position[0]), "lead_speed": float(lead.speed),
                         "lane_change_duration": float(lead.lane_change_duration_s),
                         "event_start": float(lead.event_start_s)}
                if env.vehicle.crashed or lead.crashed:
                    raise RuntimeError(f"initial overlap: {axis}={value}")
            finally:
                env.close()
            outcome, _ = run_build_episode("idm_ref", changed, changed["simulator_seed"])
            phases = {event.get("phase") for event in outcome["lead_events"]}
            if "LANE_CHANGE" not in phases or valid_label(outcome) is None:
                raise RuntimeError(f"S01 event or valid outcome missing: {axis}={value}")
            records.append({"axis": axis, "bound": name, "value": value,
                            "initial": state, "outcome": valid_label(outcome),
                            "event_times": outcome.get("event_times")})
    mapping = {row["axis"]: [r for r in records if r["axis"] == row["axis"]]
               for row in records}
    field = {"initial_clearance_m": "lead_x", "lead_speed_mps": "lead_speed",
             "lane_change_time_scale_s": "lane_change_duration", "event_start_s": "event_start"}
    for axis, pair in mapping.items():
        if pair[0]["initial"][field[axis]] == pair[1]["initial"][field[axis]]:
            raise RuntimeError(f"S01 axis does not affect runner: {axis}")
    return records


def freeze() -> None:
    config = _config()
    manifest_path = ROOT / "scenario_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError("frozen manifest already exists; never overwrite it")
    source = _jsonl(SOURCE_MANIFEST)
    count = config["candidate_count"]
    subset = [row for row in source if row.get("catalogue_id") == "S01"][:count]
    if len(subset) != count or [row["sample_index"] for row in subset] != list(range(count)):
        raise ValueError("source Sobol sequence lacks the required S01 prefix")
    smoke = _smoke(subset)
    ROOT.mkdir(parents=True, exist_ok=True)
    _json(ROOT / "environment.json", _environment())
    for row in subset:
        _append(manifest_path, row)
    code_files = sorted(MODULE_DIR.glob("fm2_*.py"))
    protocol = {"family": "S01", "target_build": TARGET, "historical_builds": list(SOURCES),
                "candidate_count": count, "budget": 50, "budget_checkpoints": list(CHECKPOINTS),
                "selector_repeats": config["selector_repeats"], "simulator_seed": subset[0]["simulator_seed"],
                "realization_rule": "same frozen seed and scenario for each build; one physical bank per build",
                "source_manifest_sha256": _hash(SOURCE_MANIFEST),
                "manifest_sha256": _hash(manifest_path), "scenario_schema_sha256": _hash(CONFIG),
                "runner_sha256": _hash(Path("highway_sim_env/envs/fbrt_unified_env.py")),
                "selector_model_source_sha256": {p.name: _hash(p) for p in code_files},
                "pilot_config_sha256": _hash(EXPERIMENT_CONFIG),
                "checkpoint_selection_rule": "minimum held-out nl3_v2 pseudo-target validation loss; tie lower seed",
                "training_seeds": config["training_seeds"],
                "previously_observed_target_prefix": config.get("previous_target_bank_prefix", 0),
                "development_continuation_not_independent_confirmation": bool(config.get("previous_target_bank_prefix")),
                "clear_advantage_rule": "FM2-FBT mean D@50 >= strongest baseline mean D@50 + 2, median D@50 higher, D@20 and D@30 no worse",
                "status": "frozen_before_target_measurement"}
    _json(ROOT / "protocol.json", protocol)
    _json(ROOT / "freeze_audit.json", {"smoke_tests": smoke,
                                       "source_count": len(source), "prefix_count": len(subset),
                                       "manifest_sha256": protocol["manifest_sha256"],
                                       "target_bank_existed_at_freeze": (ROOT / "target/full_response_bank.jsonl").exists()})


def _run_historical_build(build: str, scenarios: list[dict], output: Path,
                          prior_rows: list[dict]) -> tuple[str, int]:
    """Each worker owns one file, so interrupted full banks can resume safely."""
    if not output.exists():
        for row in prior_rows:
            _append(output, row)
    existing = _jsonl(output)
    done = {row["scenario_id"] for row in existing}
    allowed = {scene["scenario_id"] for scene in scenarios}
    if len(done) != len(existing) or not done <= allowed or any(
            row.get("build_id") != build for row in existing):
        raise ValueError(f"invalid historical partial bank: {build}")
    new_count = 0
    for scene in scenarios:
        if scene["scenario_id"] in done:
            continue
        outcome, _ = run_build_episode(build, scene, scene["simulator_seed"])
        if outcome["scenario_id"] != scene["scenario_id"]:
            raise RuntimeError(f"source execution scenario mismatch: {build}")
        outcome["visibility"] = "historical"
        _append(output, outcome)
        done.add(scene["scenario_id"])
        new_count += 1
        if new_count % 256 == 0:
            print(f"source={build} new_episodes={new_count}", flush=True)
    if len(done) != len(scenarios):
        raise RuntimeError(f"incomplete historical bank: {build}")
    return build, new_count


def _full_history(scenarios: list[dict], config: dict) -> None:
    if (ROOT / "target/full_response_bank.jsonl").exists() or list((ROOT / "target").glob("part_*.jsonl")):
        raise RuntimeError("historical bank cannot be constructed after target measurement")
    prior = _jsonl(Path(config["reuse_history_bank"]))
    if any(row.get("build_id") not in SOURCES for row in prior):
        raise ValueError("reusable historical bank contains an undeclared or target build")
    prior_by_build = {build: [row for row in prior if row["build_id"] == build]
                      for build in SOURCES}
    expected = {scene["scenario_id"] for scene in scenarios}
    if any(len({r["scenario_id"] for r in rows}) != len(rows) or
           not {r["scenario_id"] for r in rows} <= expected or
           any(r.get("build_id") != build for r in rows)
           for build, rows in prior_by_build.items()):
        raise ValueError("reused source responses do not match the frozen S01 pool")
    workers = int(config.get("history_workers", 1))
    files = {build: ROOT / "history" / f"source_{build}.jsonl" for build in SOURCES}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_historical_build, build, scenarios, files[build],
                                   prior_by_build[build]) for build in SOURCES]
        for future in as_completed(futures):
            build, count = future.result()
            print(f"source={build} completed_new_episodes={count}", flush=True)
    combined = ROOT / "history/source_response_bank.jsonl"
    if combined.exists():
        raise FileExistsError("combined historical bank already exists")
    rows = []
    for build in SOURCES:
        build_rows = _jsonl(files[build])
        if len(build_rows) != len(scenarios):
            raise RuntimeError(f"incomplete source bank: {build}")
        rows.extend(build_rows)
    for row in rows:
        _append(combined, row)
    cards = build_memory(rows, coordinates(scenarios), exclude=TARGET)
    for card in cards:
        _append(ROOT / "history/pattern_cards.jsonl", card.as_dict())
    reused = len(prior)
    _json(ROOT / "history/source_inventory.json", {
        "declared_sources": list(SOURCES), "historical_cases_per_source": len(scenarios),
        "reused_prefix_per_source": len(prior_by_build[SOURCES[0]]),
        "memory_policy": "complete measured S01 V3 bank per historical build"})
    _json(ROOT / "history/cost_ledger.json", {
        "fresh_source_episodes": len(rows) - reused,
        "reused_source_episodes": reused,
        "source_contract_smoke_episodes": 8,
        "total_pilot_source_physical_episodes": len(rows) - reused + 8,
        "by_build": dict(Counter(r["build_id"] for r in rows)),
        "valid_failures": dict(Counter(r["build_id"] for r in rows if valid_label(r) == 1)),
        "inconclusive": sum(valid_label(r) is None for r in rows),
        "pattern_count": len(cards), "pattern_cap": 64})


def history() -> None:
    scenarios = _manifest()
    _full_history(scenarios, _config())


def train_stage() -> None:
    if (ROOT / "target/full_response_bank.jsonl").exists() or list((ROOT / "target").glob("part_*.jsonl")):
        raise RuntimeError("training after target bank creation is prohibited")
    config = _config()
    scenarios = _manifest()
    rows = _historical_bank(scenarios)
    train(rows, scenarios, coordinates(scenarios), ROOT / "model", config)
    _json(ROOT / "model/config.json", config)
    protocol_path = ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["selector_model_source_sha256"] = {
        p.name: _hash(p) for p in sorted(MODULE_DIR.glob("fm2_*.py"))}
    protocol["pilot_config_sha256"] = _hash(EXPERIMENT_CONFIG)
    protocol["selected_checkpoint_sha256"] = _hash(ROOT / "model/checkpoint.pt")
    protocol["clear_advantage_rule"] = (
        "FM2-FBT mean D@50 >= strongest baseline mean D@50 + "
        f"{_config()['clear_advantage_min_d50_gain']}, median D@50 higher, D@20 and D@30 no worse")
    protocol["status"] = "checkpoint_selected_before_target_measurement"
    _json(protocol_path, protocol)


def _run_target_chunk(scenarios: list[dict], output: Path) -> int:
    existing = _jsonl(output) if output.exists() else []
    done = {row["scenario_id"] for row in existing}
    allowed = {scene["scenario_id"] for scene in scenarios}
    if len(done) != len(existing) or not done <= allowed or any(
            row.get("build_id") != TARGET for row in existing):
        raise ValueError(f"invalid target partial bank: {output}")
    new_count = 0
    for scene in scenarios:
        if scene["scenario_id"] in done:
            continue
        outcome, _ = run_build_episode(TARGET, scene, scene["simulator_seed"])
        outcome["visibility"] = "evaluator_only"
        _append(output, outcome)
        done.add(scene["scenario_id"])
        new_count += 1
        if new_count % 128 == 0:
            print(f"target_chunk={output.stem} new_episodes={new_count}", flush=True)
    if len(done) != len(scenarios):
        raise RuntimeError(f"target chunk incomplete: {output}")
    return new_count


def target_bank() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("selected_checkpoint_sha256") != _hash(ROOT / "model/checkpoint.pt"):
        raise RuntimeError("selected checkpoint not frozen before target measurement")
    scenarios = _manifest()
    path = ROOT / "target/full_response_bank.jsonl"
    config = _config()
    prior_path = config.get("reuse_target_bank_prefix")
    prior_ids = set()
    prior_rows = []
    if prior_path:
        prior = Path(prior_path)
        prior_rows = _jsonl(prior)
        prefix = int(config["previous_target_bank_prefix"])
        expected = [scene["scenario_id"] for scene in scenarios[:prefix]]
        if [row["scenario_id"] for row in prior_rows] != expected or any(
                row.get("build_id") != TARGET for row in prior_rows):
            raise ValueError("prior target bank does not match frozen S01 prefix and target build")
        prior_ids = set(expected)
    workers = int(config.get("target_workers", 0))
    if workers and not path.exists():
        pending = scenarios[len(prior_rows):]
        chunks = [pending[i::workers] for i in range(workers)]
        files = [ROOT / "target" / f"part_{i:02d}.jsonl" for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_target_chunk, chunk, output)
                       for chunk, output in zip(chunks, files)]
            for future in as_completed(futures):
                print(f"target_chunk_completed_new_episodes={future.result()}", flush=True)
        parts = {row["scenario_id"]: row for output in files for row in _jsonl(output)}
        if len(parts) != len(pending) or set(parts) != {scene["scenario_id"] for scene in pending}:
            raise RuntimeError("target chunks do not cover the fixed pool")
        for row in prior_rows:
            _append(path, row)
        for scene in pending:
            _append(path, parts[scene["scenario_id"]])
    elif prior_path and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(prior_path).read_bytes())
    existing = _jsonl(path) if path.exists() else []
    done = {r["scenario_id"] for r in existing}
    if len(done) != len(existing):
        raise ValueError("duplicate target physical response")
    for scene in scenarios:
        if scene["scenario_id"] in done:
            continue
        outcome, _ = run_build_episode(TARGET, scene, scene["simulator_seed"])
        outcome["visibility"] = "evaluator_only"
        _append(path, outcome)
        done.add(scene["scenario_id"])
    if len(done) != _config()["candidate_count"]:
        raise RuntimeError("target response bank incomplete")
    bank_rows = list(_target_response_bank(scenarios).values())
    truth_path = ROOT / "target/failure_truth.jsonl"
    if not truth_path.exists():
        for row in bank_rows:
            if valid_label(row) == 1:
                _append(truth_path, row)
    truth_ids = {row["scenario_id"] for row in _jsonl(truth_path)} if truth_path.exists() else set()
    if truth_ids != {row["scenario_id"] for row in bank_rows if valid_label(row) == 1}:
        raise RuntimeError("target failure truth does not match full response bank")
    _json(ROOT / "target/cost_ledger.json", {"physical_target_episodes": len(done),
          "reused_prior_target_episodes": len(prior_ids),
          "new_target_episodes": len(done) - len(prior_ids),
          "reused_bank_path": prior_path,
          "confirmed_failure_truth_count": len(truth_ids),
          "inconclusive_count": sum(valid_label(row) is None for row in bank_rows),
          "target_query_budget_per_selector": 50,
          "logical_repeats_are_not_independent_physical_episodes": True})


def _model() -> FM2Model:
    config = _config()
    checkpoint = torch.load(ROOT / "model/checkpoint.pt", map_location="cpu", weights_only=False)
    model = FM2Model(width=config["width"], dropout=config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def _exact_history_rank_scores(scenarios: list[dict], history: list[dict]) -> np.ndarray:
    """The legacy percentile rule evaluated once per measured source row."""
    ids = [scene["scenario_id"] for scene in scenarios]
    scores = np.zeros(len(ids), dtype=float)
    for build in SOURCES:
        rows = [row for row in history if row.get("build_id") == build
                and row.get("template_id") == "fbrt_cutin"]
        if len(rows) != len(ids) or {row["scenario_id"] for row in rows} != set(ids) or any(
                row.get("inconclusive") for row in rows):
            # The fast path is exact only for one valid measurement per case.
            return np.asarray([_history_rank(scene, history) for scene in scenarios], dtype=float)
        risks = {}
        for row in rows:
            collision = 1.0 if row.get("ego_collision") is True else 0.0
            ttc = row.get("min_ttc")
            clearance = row.get("min_clearance")
            risks[row["scenario_id"]] = (collision +
                0.49 / (1 + max(0.0, float(ttc or 1e6))) +
                0.01 / (1 + max(0.0, float(clearance or 1e6))))
        sorted_risks = np.sort(np.fromiter(risks.values(), dtype=float))
        scores += np.asarray([np.searchsorted(sorted_risks, risks[sid], side="right")
                              / len(sorted_risks) for sid in ids]) / len(SOURCES)
    return scores


def _failure_distance_replay(scenarios: list[dict], history: list[dict],
                             oracle: TargetOracle, seed: int, budget: int = 50) -> list[dict]:
    """Vectorized single-family replay of the existing FailureDistance-v2 rule."""
    if any(scene["template_id"] != "fbrt_cutin" for scene in scenarios):
        raise ValueError("FailureDistance adapter is only defined for S01")
    x = np.asarray([active_values(scene) for scene in scenarios], dtype=float)
    failed = np.asarray([active_values(row["scenario"]) for row in history
                         if row.get("template_id") == "fbrt_cutin" and
                         row.get("ego_collision") is True and not row.get("inconclusive", False)],
                        dtype=float).reshape(-1, x.shape[1])
    distances = cdist(x, failed).min(axis=1) if len(failed) else np.full(len(x), 2.0)
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(x), dtype=bool)
    selected: list[int] = []
    queries = []
    for rank in range(1, min(budget, len(x)) + 1):
        available = np.flatnonzero(remaining)
        if rank in (10, 20):
            art = cdist(x[available], x[selected]).min(axis=1) if selected else np.full(len(available), 2.0)
            best = art.max()
            tied = available[np.isclose(art, best)]
            reason = "global_art_maximin"
        else:
            current = distances[available]
            best = current.min()
            tied = available[np.isclose(current, best)]
            reason = "nearest_observed_failure"
        chosen = int(tied[int(rng.integers(len(tied)))])
        outcome = oracle.query(scenarios[chosen]["scenario_id"])
        reward = int(valid_label(outcome) == 1)
        remaining[chosen] = False
        selected.append(chosen)
        if reward:
            distances = np.minimum(distances, np.linalg.norm(x - x[chosen], axis=1))
        queries.append({"method": "FailureDistance-v2", "rank": rank,
                        "scenario_id": scenarios[chosen]["scenario_id"],
                        "selection_reason": reason, "valid_collision": bool(reward),
                        "selection_reward": reward,
                        "ego_collision": outcome.get("ego_collision"),
                        "completed": outcome.get("completed"),
                        "inconclusive": outcome.get("inconclusive"),
                        "episode_cost": 1, "execution_id": outcome.get("execution_id")})
    return queries


def evaluate(*, with_dfe: bool) -> None:
    scenarios = _manifest()
    coords = coordinates(scenarios)
    source = _historical_bank(scenarios)
    cards = build_memory(source, coords, exclude=TARGET)
    memory = memory_arrays(cards, source)
    bank = _target_response_bank(scenarios)
    if not any(valid_label(row) == 1 for row in bank.values()):
        print("zero-failure target bank; selector comparison skipped", flush=True)
        return
    evaluation = ROOT / "evaluation"
    evaluation.mkdir(exist_ok=True)
    config = _config()
    methods = ("FM2-FBT",) if with_dfe else ("FM2-NoDFE", *BASELINES)
    model = _model()
    history_rank_scores = None
    for method in methods:
        if method == "HistoryRank-UCB-v2":
            # With one template the UCB arm cannot change. The historical score
            # is fixed, so caching it preserves the old selection rule exactly.
            history_rank_scores = _exact_history_rank_scores(scenarios, source)
            _json(evaluation / "historyrank_score_cache.json", {
                "manifest_sha256": _hash(ROOT / "scenario_manifest.jsonl"),
                "source_bank_sha256": _hash(ROOT / "history/source_response_bank.jsonl"),
                "scores": history_rank_scores.tolist()})
        for repeat in range(config["selector_repeats"]):
            suffix = f"{method.replace('/', '_')}_{repeat:02d}"
            output = evaluation / f"queries_{suffix}.jsonl"
            if output.exists():
                if len(_jsonl(output)) != 50:
                    raise RuntimeError(f"incomplete query file: {output}")
                continue
            oracle = TargetOracle(bank)
            if method.startswith("FM2-"):
                queries, retrieval, router = run_fm2(
                    model, scenarios, coords, memory, cards, oracle, budget=50,
                    seed=int(config.get("selector_seed", 4179950)) + repeat, dfe_enabled=with_dfe,
                    dfe_config=config["dfe"], router_discount=config["router_discount"])
                for row in retrieval:
                    _append(evaluation / f"retrieval_audit_{suffix}.jsonl", row)
                for row in router:
                    _append(evaluation / f"router_audit_{suffix}.jsonl", row)
            elif method == "HistoryRank-UCB-v2":
                assert history_rank_scores is not None
                rng = np.random.default_rng(int(config.get("selector_seed", 4179950)) + repeat)
                remaining = set(range(len(scenarios)))
                queries = []
                for rank in range(1, 51):
                    best = max(history_rank_scores[i] for i in remaining)
                    tied = [i for i in sorted(remaining) if np.isclose(history_rank_scores[i], best)]
                    chosen = tied[int(rng.integers(len(tied)))]
                    outcome = oracle.query(scenarios[chosen]["scenario_id"])
                    remaining.remove(chosen)
                    reward = int(valid_label(outcome) == 1)
                    queries.append({"method": method, "rank": rank,
                                    "scenario_id": scenarios[chosen]["scenario_id"],
                                    "selection_reason": "history_risk_percentile_ucb_single_family_cached",
                                    "history_score": float(history_rank_scores[chosen]),
                                    "valid_collision": bool(reward),
                                    "selection_reward": reward,
                                    "ego_collision": outcome.get("ego_collision"),
                                    "completed": outcome.get("completed"),
                                    "inconclusive": outcome.get("inconclusive"),
                                    "episode_cost": 1,
                                    "execution_id": outcome.get("execution_id")})
            elif method == "FailureDistance-v2":
                queries = _failure_distance_replay(
                    scenarios, source, oracle, int(config.get("selector_seed", 4179950)) + repeat)
            else:
                queries, _, _, _ = run_selector(method, scenarios, source, oracle,
                                                budget=50, random_seed=int(config.get("selector_seed", 4179950)) + repeat,
                                                target_build_id=TARGET, mode="cross_agent",
                                                session_id=f"fm2_s01:{method}:{repeat}",
                                                coverage_slots=0 if method == "FBRT-Memory-Exploit-v3" else None)
            if len(queries) != 50 or len(oracle.queried) != 50:
                raise RuntimeError(f"method {method} did not spend exactly 50 queries")
            for row in queries:
                _append(output, row)


def report() -> None:
    eval_dir = ROOT / "evaluation"
    scenarios = _manifest()
    _historical_bank(scenarios)
    bank = list(_target_response_bank(scenarios).values())
    truth = {row["scenario_id"] for row in bank if valid_label(row) == 1}
    files = sorted(eval_dir.glob("queries_*.jsonl"))
    if not files:
        if not any(valid_label(row) == 1 for row in bank):
            (ROOT / "report.md").write_text(
                f"# FM²-FBT S01 development pilot\n\nGT target failures: 0/{len(bank)}. "
                "This frozen task cannot compare failure discovery. "
                "A different target requires a new development protocol.\n", encoding="utf-8")
            return
        raise RuntimeError("no selector outputs")
    gt = sum(valid_label(row) == 1 for row in bank)
    rows, repeats = [], []
    curves: dict[str, list[np.ndarray]] = {}
    for path in files:
        queries = _jsonl(path)
        if len(queries) != 50:
            raise RuntimeError(f"incomplete selector output: {path}")
        if len({row["scenario_id"] for row in queries}) != 50 or any(
                bool(row["valid_collision"]) != (row["scenario_id"] in truth) for row in queries):
            raise RuntimeError(f"selector discoveries disagree with target truth: {path}")
        discovered = {row["scenario_id"] for row in queries if row["scenario_id"] in truth}
        _json(eval_dir / f"truth_coverage_{path.stem.removeprefix('queries_')}.json", {
            "method": queries[0]["method"], "truth_count": len(truth),
            "discovered_failure_ids": sorted(discovered),
            "unfound_failure_ids": sorted(truth - discovered),
            "recall_at_50": len(discovered) / len(truth) if truth else None})
        method = queries[0]["method"]
        failures = np.cumsum([int(row["valid_collision"]) for row in queries])
        curves.setdefault(method, []).append(failures)
        summary = {"method": method, "repeat": len(curves[method]) - 1,
                   **{f"D@{k}": int(failures[k-1]) for k in CHECKPOINTS},
                   "Recall@50": float(failures[-1] / gt) if gt else None,
                   "HitRate@50": float(failures[-1] / 50),
                   "UnfoundTruth@50": gt - int(failures[-1]),
                   "first_failure_rank": next((i+1 for i, x in enumerate(failures) if x >= 1), None),
                   "queries_until_5_failures": next((i+1 for i, x in enumerate(failures) if x >= 5), None),
                   "global_queries": sum(row.get("selected_arm") == "global" for row in queries),
                   "local_queries": sum(row.get("selected_arm") == "local" for row in queries),
                   "global_failures": sum(row.get("selected_arm") == "global" and row["valid_collision"] for row in queries),
                   "local_failures": sum(row.get("selected_arm") == "local" and row["valid_collision"] for row in queries)}
        repeats.append(summary)
    for method, values in sorted(curves.items()):
        subset = [row for row in repeats if row["method"] == method]
        aggregate = {"method": method, "repeats": len(subset)}
        for key in [f"D@{k}" for k in CHECKPOINTS] + ["Recall@50", "HitRate@50", "UnfoundTruth@50"]:
            numeric = [row[key] for row in subset if row[key] is not None]
            aggregate[key] = float(np.mean(numeric)) if numeric else None
            aggregate[key + "_median"] = float(np.median(numeric)) if numeric else None
            aggregate[key + "_std"] = float(np.std(numeric)) if numeric else None
        rows.append(aggregate)
    for name, data in (("repeats.csv", repeats), ("summary.csv", rows)):
        with (eval_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, values in sorted(curves.items()):
        y = np.mean(values, axis=0)
        ax.plot(range(1, 51), y, label=method)
    ax.set(xlabel="Target queries", ylabel="Cumulative confirmed failures", xlim=(1, 50))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(eval_dir / "figure_failures_vs_budget.png", dpi=180)
    plt.close(fig)
    by_method = {row["method"]: row for row in rows}
    strongest = max((row for row in rows if not row["method"].startswith("FM2-")),
                    key=lambda row: (row["D@50"], row["D@20"], row["D@30"],
                                     row["D@10"], row["D@5"]), default=None)
    fm = by_method.get("FM2-FBT")
    no = by_method.get("FM2-NoDFE")
    retrieval_files = sorted(eval_dir.glob("retrieval_audit_FM2-FBT_*.jsonl"))
    probe_shifts = []
    for file in retrieval_files:
        audit_rows = _jsonl(file)
        if len(audit_rows) > 1:
            start = np.asarray(audit_rows[0]["fixed_probe_weights"], dtype=float)
            end = np.asarray(audit_rows[-1]["fixed_probe_weights"], dtype=float)
            probe_shifts.append(float(np.abs(start - end).sum()))
    probe_shift = float(np.mean(probe_shifts)) if probe_shifts else None
    fbt_repeats = [row for row in repeats if row["method"] == "FM2-FBT"]
    router_yields = {key: float(np.mean([row[key] for row in fbt_repeats]))
                     for key in ("global_queries", "local_queries", "global_failures", "local_failures")}
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    source_rows = _jsonl(ROOT / "history/source_response_bank.jsonl")
    manifest_ids = {row["scenario_id"] for row in _manifest()}
    leakage_audit = {
        "source_target_excluded": all(row["build_id"] != TARGET for row in source_rows),
        "model_checkpoint_matches_pre_target_protocol": protocol.get("selected_checkpoint_sha256") ==
        _hash(ROOT / "model/checkpoint.pt"),
        "model_checkpoint_predates_target_bank": (ROOT / "model/checkpoint.pt").stat().st_mtime <
        (ROOT / "target/full_response_bank.jsonl").stat().st_mtime,
        "all_queries_in_frozen_manifest": all(set(q["scenario_id"] for q in _jsonl(path)) <= manifest_ids
                                              for path in files),
        "all_sessions_have_50_distinct_queries": all(len(q := _jsonl(path)) == 50 and
                                                       len({r["scenario_id"] for r in q}) == 50
                                                       for path in files),
        "protocol_amendments": str(ROOT / "protocol_amendments.jsonl")}
    _json(eval_dir / "leakage_audit.json", leakage_audit)
    table = ["| Method | D@5 | D@10 | D@20 | D@30 | D@50 | Recall@50 | HitRate@50 | Unfound truth@50 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append("| " + row["method"] + " | " + " | ".join(
            "NA" if row[key] is None else f"{row[key]:.3f}" for key in
            ["D@5", "D@10", "D@20", "D@30", "D@50", "Recall@50", "HitRate@50", "UnfoundTruth@50"]) + " |")
    clear = bool(fm and strongest and
                 fm["D@50"] >= strongest["D@50"] + _config()["clear_advantage_min_d50_gain"] and
                 fm["D@50_median"] > strongest["D@50_median"] and
                 fm["D@20"] >= strongest["D@20"] and fm["D@30"] >= strongest["D@30"])
    conclusion = ("冻结候选池无目标 failure，无法比较发现能力。"
                  if gt == 0 else
                  "FM2-FBT 在本 S01 开发库中达到预先规定的清晰优势判据。"
                  if clear else
                  "FM2-FBT 在本 S01 开发库中未达到预先规定的清晰优势判据。")
    history_cost = json.loads((ROOT / "history/cost_ledger.json").read_text(encoding="utf-8"))
    target_cost = json.loads((ROOT / "target/cost_ledger.json").read_text(encoding="utf-8"))
    training_summary = json.loads((ROOT / "model/validation_summary.json").read_text(encoding="utf-8"))
    train_seconds = sum(row["training_wall_seconds"] for row in training_summary["seed_results"])
    report_text = ["# FM²-FBT S01 development pilot", "", f"冻结候选池中的目标真实 failure：{gt}/{len(bank)}。",
                   "", *table, "", conclusion, "",
                   f"最强既有基线（先按 D@50，平局再按 D@20/D@30）：{strongest['method'] if strongest else 'NA'}。",
                   f"FM2-NoDFE 的 D@50 为 {no['D@50'] if no else 'NA'}；FM2-FBT 为 {fm['D@50'] if fm else 'NA'}。",
                   (f"DFE 相对 NoDFE：D@20 {fm['D@20']-no['D@20']:+.2f}，"
                    f"D@30 {fm['D@30']-no['D@30']:+.2f}，D@50 {fm['D@50']-no['D@50']:+.2f}。")
                   if fm and no else "",
                   f"固定候选的历史检索权重在目标反馈后的平均 L1 变化：{probe_shift if probe_shift is not None else 'NA'}。",
                   ("FM2-FBT 每次会话平均：Global 查询 {global_queries:.1f} 次、发现 {global_failures:.1f} 个；"
                    "Local 查询 {local_queries:.1f} 次、发现 {local_failures:.1f} 个。".format(**router_yields)),
                    "首次 failure 位次和发现 5 个 failure 所需查询数见 evaluation/repeats.csv；"
                    "本轮每方法只有一个选择种子，不估计跨种子方差。",
                   "", "## 诊断与边界", "",
                   (f"该库有 {gt} 个目标 failure；50 次预算内最多可发现 {min(gt, 50)} 个。"
                    "结果仅支持当前固定库和目标版本的开发判断。"),
                   ("Component 2 的目标证据作用可结合固定候选检索权重变化及 NoDFE 曲线判断；"
                    "Component 3 的作用以 FBT 与 NoDFE 的各预算截点差值判断。"),
                   ("本库是开发续测，不能作为独立 confirmation。"
                    if protocol.get("development_continuation_not_independent_confirmation") else
                    "本库不得作为独立 confirmation。"),
                   "", "## 信息隔离与成本", "",
                   f"泄漏审计：{leakage_audit}。",
                   (f"物理执行：历史响应库 {len(source_rows)} 次（本轮新测 {history_cost['fresh_source_episodes']} 次），"
                    "source 契约冒烟测试 "
                    f"{history_cost['source_contract_smoke_episodes']} 次，target bank "
                    f"{target_cost['physical_target_episodes']} 次（本轮新测 {target_cost.get('new_target_episodes', target_cost['physical_target_episodes'])} 次）；"
                    f"{len(training_summary['seed_results'])} 个训练种子累计约 {train_seconds:.1f} 秒。"),
                   f"每方法 {protocol['selector_repeats']} 次逻辑重放共用一份物理目标响应库，不是独立车辆试验。",
                   "Attention 权重仅作检索诊断，不作因果解释。",
                   "目标结果仅经计费的 TargetOracle.query 进入选择器。"]
    (ROOT / "report.md").write_text("\n".join(report_text) + "\n", encoding="utf-8")


def main() -> None:
    global ROOT, EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("freeze", "history", "train", "bank", "evaluate_a",
                                             "evaluate_b", "report", "all"), default="all")
    args = parser.parse_args()
    ROOT, EXPERIMENT_CONFIG = args.root, args.config
    stage = args.stage
    steps = (("freeze", freeze), ("history", history), ("train", train_stage),
             ("bank", target_bank), ("evaluate_a", lambda: evaluate(with_dfe=False)),
             ("evaluate_b", lambda: evaluate(with_dfe=True)), ("report", report))
    for name, action in steps:
        if stage in (name, "all"):
            print(f"stage={name}", flush=True)
            action()


if __name__ == "__main__":
    main()
