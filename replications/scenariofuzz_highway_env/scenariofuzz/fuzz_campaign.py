"""Run real online ScenarioFuzz-H and its four attributable ablations."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from sut_algorithms.highway_env.idm_profiles import get_profile

from .corpus import ScenarioSpec, build_default_corpus, save_corpus
from .driving_score import margin_score
from .execution import execute_scenario
from .filter import load_checkpoint, predict_scores, select_candidates
from .io_utils import append_jsonl, file_hash, load_config, write_csv
from .mutators import generate_candidates
from .scheduler import FrequencyScheduler


def _variant_flags(name: str) -> tuple[bool, bool]:
    if name not in {"RMS-H", "2SMS-H", "RMS+SEM-H", "2SMS+SEM-H"}:
        raise ValueError(name)
    return name.startswith("2SMS"), "+SEM" in name


def run_campaign(config_path: Path, sem_path: Path, output: Path) -> None:
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    seeds = build_default_corpus(config)
    save_corpus(output / "seed_corpus.json", seeds)
    shutil.copyfile(config_path, output / "config.resolved.yaml")
    model, checkpoint = load_checkpoint(sem_path, "cuda" if __import__("torch").cuda.is_available() else "cpu")
    device = next(model.parameters()).device.type
    profile = get_profile(config["target_sut"])
    all_summary: list[dict] = []
    started_all = time.perf_counter()
    for protocol_index, protocol in enumerate(config["protocols"]):
        for variant_index, variant in enumerate(config["variants"]):
            two_stage, use_sem = _variant_flags(variant)
            rng = np.random.default_rng(int(config["random_seed"]) + 1000 * protocol_index + 100 * variant_index)
            scheduler = FrequencyScheduler(tuple(seeds), rng)
            executions = 0
            collisions = critical = invalid = empty_filters = 0
            proposed = prechecked = unique = sem_pass = 0
            candidate_generation_seconds = sem_inference_seconds = 0.0
            simulation_wall_seconds = simulation_seconds = 0.0
            executed_scenario_ids: set[str] = set()
            duplicate_executions = 0
            first_event = None
            batch_counter = 0
            cycle = 0
            reference: ScenarioSpec | None = None
            variant_records: list[dict] = []
            variant_started = time.perf_counter()
            while executions < int(config["budget"]) and batch_counter < int(config["max_candidate_batches"]) * int(config["budget"]):
                seed = scheduler.choose()
                cycle = cycle + 1 if reference is not None and cycle < int(config["Nc"]) else 1
                if cycle == 1:
                    reference = None
                nm = int(config["Nm"])
                if protocol == "paper_nm" and not use_sem:
                    nm = int(config["Ne"])
                candidate_started = time.perf_counter()
                batch = generate_candidates(
                    seed, nm, rng,
                    reference=reference if two_stage and cycle > 1 else None,
                    gap_step=float(config["neighbor_step"]["initial_gap"]),
                    speed_step=float(config["neighbor_step"]["relative_speed"]),
                )
                candidate_generation_seconds += time.perf_counter() - candidate_started
                batch_counter += 1
                proposed += batch.attempted
                prechecked += batch.attempted - len(batch.rejected)
                unique += len(batch.candidates)
                candidates = list(batch.candidates)
                inference_started = time.perf_counter()
                scores = predict_scores(model, seed, candidates, device) if use_sem and candidates else None
                sem_inference_seconds += time.perf_counter() - inference_started
                if scores is not None:
                    sem_pass += int(np.sum(scores > float(config["filter_threshold"])))
                selected, reason = select_candidates(candidates, scores, int(config["Ne"]), float(config["filter_threshold"]), rng, config["empty_filter_policy"])
                if not selected:
                    empty_filters += 1
                candidate_rows = []
                selected_ids = {candidates[index].scenario_id for index in selected}
                for i, candidate in enumerate(candidates):
                    candidate_rows.append({**candidate.to_dict(), "predicted_score": None if scores is None else float(scores[i]), "selected": candidate.scenario_id in selected_ids})
                append_jsonl(output / "candidate_batches.jsonl", {
                    "protocol": protocol, "variant": variant, "batch": batch_counter, "cycle": cycle, "stage": batch.stage,
                    "seed_id": seed.seed_id, "reference_id": batch.reference_id, "attempted": batch.attempted, "rejected": list(batch.rejected),
                    "duplicates": batch.duplicates, "selection_reason": reason, "candidates": candidate_rows,
                })
                safe_references: list[tuple[float, ScenarioSpec]] = []
                failed_seed = False
                for candidate_index in selected:
                    if executions >= int(config["budget"]):
                        break
                    candidate = candidates[candidate_index]
                    score = None if scores is None else float(scores[candidate_index])
                    observation = execute_scenario(profile, candidate, int(config["random_seed"]) + executions, output / "trajectories")
                    executions += 1
                    duplicate_executions += int(candidate.scenario_id in executed_scenario_ids)
                    executed_scenario_ids.add(candidate.scenario_id)
                    simulation_wall_seconds += observation.wall_seconds
                    simulation_seconds += observation.simulation_seconds
                    event = bool(observation.collision)
                    critical_event = bool(observation.collision or observation.near_miss)
                    collisions += int(event); critical += int(critical_event); invalid += int(not observation.valid)
                    if event and first_event is None:
                        first_event = executions
                    row = {
                        "protocol": protocol, "variant": variant, "step": executions, "cycle": cycle,
                        **candidate.to_dict(), "target_sut": profile.name, "predicted_score": score,
                        "score_semantics": "SEM historical failure tendency" if use_sem else "not_scored_random_selection",
                        "visible_target_count": executions - 1, "history_split_id": "source_only_grouped_v1",
                        "selected_reason": reason, "observed_event": event, "critical_event": critical_event,
                        "collision": observation.collision, "near_miss": observation.near_miss,
                        "vulnerability": observation.vulnerability, "driving_score": margin_score(observation.vulnerability),
                        "valid": observation.valid, "invalid_reason": observation.invalid_reason,
                        "cumulative_cost": executions, "simulation_seconds": observation.simulation_seconds,
                        "wall_seconds": observation.wall_seconds, "execution_id": observation.execution_id,
                        "trajectory_path": observation.trajectory_path,
                    }
                    append_jsonl(output / "selection_log.jsonl", row)
                    append_jsonl(output / "executions.jsonl", row)
                    variant_records.append(row)
                    if observation.valid and not event:
                        safe_references.append((row["driving_score"], candidate))
                    if event:
                        failed_seed = True
                        break
                if failed_seed or cycle >= int(config["Nc"]):
                    reference = None; cycle = 0
                elif safe_references and two_stage:
                    reference = min(safe_references, key=lambda item: item[0])[1]
                elif two_stage:
                    reference = None; cycle = 0
            all_summary.append({
                "protocol": protocol, "variant": variant, "target_sut": profile.name,
                "executions": executions, "collisions": collisions, "critical_events": critical,
                "first_collision_budget": first_event if first_event is not None else "censored",
                "first_collision_censored": first_event is None,
                "invalid_executions": invalid, "empty_filters": empty_filters, "candidate_attempts": proposed,
                "precheck_passed": prechecked, "unique_candidates": unique, "sem_above_threshold": sem_pass,
                "duplicate_executions": duplicate_executions,
                "duplicate_execution_ratio": duplicate_executions / max(executions, 1),
                "simulation_seconds": simulation_seconds,
                "simulation_wall_seconds": simulation_wall_seconds,
                "candidate_generation_seconds": candidate_generation_seconds,
                "sem_inference_seconds": sem_inference_seconds,
                "collisions_per_simulation_minute": collisions / max(simulation_seconds / 60.0, 1e-12),
                "wall_seconds": time.perf_counter() - variant_started,
            })
    write_csv(output / "ablation_summary.csv", all_summary)
    trajectory_rows = []
    for line in (output / "executions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        trajectory_rows.append({key: row[key] for key in ("execution_id", "scenario_id", "protocol", "variant", "mode", "collision", "near_miss", "trajectory_path")})
    (output / "trajectory_manifest.json").write_text(json.dumps(trajectory_rows, indent=2), encoding="utf-8")
    manifest = {
        "method": "ScenarioFuzz-H", "task_kind": config["task_kind"], "target_sut": profile.name,
        "sem_checkpoint": sem_path.as_posix(), "sem_checkpoint_sha256": file_hash(sem_path),
        "bounds_hash": seeds[0].bounds_hash(), "environment": "highway-env==1.9.1",
        "oracle": "collision; critical=collision_or_near_miss", "random_seed": config["random_seed"],
        "total_actual_target_executions": int(sum(row["executions"] for row in all_summary)),
        "total_wall_seconds": time.perf_counter() - started_all, "device": device,
        "cost_breakdown_seconds": {
            "candidate_generation": float(sum(row["candidate_generation_seconds"] for row in all_summary)),
            "sem_inference": float(sum(row["sem_inference_seconds"] for row in all_summary)),
            "simulation_wall": float(sum(row["simulation_wall_seconds"] for row in all_summary)),
            "simulated_time": float(sum(row["simulation_seconds"] for row in all_summary)),
        },
        "rejected_candidates_are_unlabelled": True, "online_and_pool_protocols_mixed": False,
        "weather_and_appearance_mutators": "disabled_no_physical_effect",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_campaign(args.config, args.sem, args.output)
    print(f"Wrote real online ScenarioFuzz-H campaigns to {args.output}")


if __name__ == "__main__":
    main()
