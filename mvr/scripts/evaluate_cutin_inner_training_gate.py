"""Evaluate whether a Cut-in Inner checkpoint has learned an executable policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..experiments.cutin_inner import expand_cutin_training_domains
from ..failure.criteria import FailureCriteria
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.parameter_space import NormalizedScenarioAction
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
    selected_tasks,
)
from ..training.trainers import build_online


def _training_tasks(config: dict[str, Any], taskbook: Path) -> list[Any]:
    cutin = config["cutin_inner"]
    tasks = selected_tasks(config, taskbook, "train", "train", "train")
    tasks = [
        task for task in tasks
        if task.sut_ref in set(cutin["training_sut_refs"])
        and task.geometry_id in set(cutin["training_geometry_ids"])
    ]
    return expand_cutin_training_domains(tasks, cutin["training_logical_domains"])


def _centre_action(task: Any) -> NormalizedScenarioAction:
    return NormalizedScenarioAction(
        0,
        tuple(
            0.5 * (float(lower) + float(upper))
            for lower, upper in task.logical_domain_bounds.values()
        ),
    )


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the Stage 1 shared-policy gate to each Logical Domain."""
    if not rows:
        raise ValueError("training gate requires at least one evaluation record")
    gate = {
        "minimum_valid_rate_per_domain": 0.75,
        "minimum_valid_event_count_per_domain": 1,
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["logical_domain_id"]), []).append(row)

    domains = {}
    for domain, domain_rows in sorted(grouped.items()):
        valid_count = sum(bool(row["valid"]) for row in domain_rows)
        event_count = sum(bool(row["event"]) for row in domain_rows)
        summary = {
            "cases": len(domain_rows),
            "valid_count": valid_count,
            "event_count": event_count,
            "valid_rate": _rate(domain_rows, "valid"),
            "event_rate": _rate(domain_rows, "event"),
            "cutin_path_or_event_rate": _rate(domain_rows, "cutin_path_or_event"),
        }
        summary["passed"] = bool(
            summary["valid_rate"] >= gate["minimum_valid_rate_per_domain"]
            and event_count >= gate["minimum_valid_event_count_per_domain"]
        )
        domains[domain] = summary

    passed_domain_count = sum(summary["passed"] for summary in domains.values())
    total_domain_count = len(domains)
    passed = passed_domain_count == total_domain_count
    return {
        "gate": gate,
        "summary": {
            "valid_rate": _rate(rows, "valid"),
            "cutin_path_or_event_rate": _rate(rows, "cutin_path_or_event"),
            "valid_event_rate": _rate(rows, "event"),
        },
        "domains": domains,
        "passed_domain_count": passed_domain_count,
        "total_domain_count": total_domain_count,
        "status": "passed" if passed else (
            "partially_passed" if passed_domain_count else "failed"
        ),
        "passed": passed,
    }


def run(config_path: str, checkpoint_path: str) -> dict[str, Any]:
    config, taskbook, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    assert_taskbook_compatible(checkpoint, taskbook)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    criteria = FailureCriteria.from_config(config["failure"])
    rows = []
    for task in _training_tasks(config, taskbook):
        action = _centre_action(task)
        for seed in (11, 22, 33):
            result = build_online(
                model, task, int(config["training"]["step_budget"]), criteria,
            ).run(
                task,
                1,
                deterministic=True,
                posterior_support_limit=0,
                scene_action_provider=lambda *_args, value=action: value,
                episode_seed_provider=lambda current, _index, value=seed: (
                    current.geometry_seed + 10_000 * value
                ),
            )
            episode = result.episodes[0]
            transitions = episode.rollout.transitions
            rows.append({
                "task_id": task.task_id,
                "logical_domain_id": task.logical_domain_id,
                "sut_ref": task.sut_ref,
                "geometry_id": task.geometry_id,
                "seed": seed,
                "valid": bool(episode.outcome["is_valid_episode"]),
                "event": bool(
                    episode.outcome["valid_target_collision"]
                    or episode.outcome["valid_critical_near_miss"]
                ),
                "cutin_completed": any(
                    bool(row["info"].get("semantic_maneuver_completed", False))
                    for row in transitions
                ),
                "minimum_target_lateral_m": min(
                    abs(float(row["info"].get("traffic_cutin_lateral_m", np.inf)))
                    for row in transitions
                ),
                "min_ttc": float(episode.outcome["min_ttc"]),
                "min_distance": float(episode.outcome["min_distance"]),
                "termination_reason": episode.outcome["termination_reason"],
                "raw_near_miss_candidate_steps": sum(
                    bool(row["info"].get("raw_near_miss_candidate", False))
                    for row in transitions
                ),
                "valid_event_capture_steps": sum(
                    bool(row["info"].get("event_just_captured", False))
                    and (
                        bool(row["info"].get("valid_target_collision", False))
                        or bool(row["info"].get(
                            "valid_critical_near_miss", False
                        ))
                    )
                    for row in transitions
                ),
            })

    for row in rows:
        # A valid event terminates the simulator by contract.  Requiring a
        # later reference endpoint after that terminal interaction would
        # reject the intended successful Cut-in behavior without changing
        # any failure or semantic criterion.
        row["cutin_path_or_event"] = bool(row["cutin_completed"] or row["event"])

    report = summarize_records(rows)
    return {
        "scope": {
            "functional_scenario": "cutin",
            "split": "train_only_policy_gate",
            "outer_trained": False,
            "test_split_accessed": False,
        },
        "checkpoint_stage": checkpoint.stage,
        **report,
        "records": rows,
    }


def paired_cases(config: dict[str, Any], taskbook: Path) -> list[dict[str, Any]]:
    """Pre-register three reproducible in-domain cases for every Logical Domain."""
    cases = []
    candidate_count = len(mvr_parameter_spaces()["cutin"].candidates)
    for domain_index, task in enumerate(_training_tasks(config, taskbook)):
        bounds = np.asarray(list(task.logical_domain_bounds.values()), dtype=float)
        samples = [
            ("centre", 0, 0.5 * (bounds[:, 0] + bounds[:, 1])),
        ]
        generator = np.random.default_rng(20260906 + domain_index)
        for sample_index, candidate in enumerate((0, 1), start=1):
            samples.append((
                f"domain_sample_{sample_index}",
                candidate % candidate_count,
                generator.uniform(bounds[:, 0], bounds[:, 1]),
            ))
        for source, candidate_index, continuous in samples:
            case_index = len(cases)
            cases.append({
                "case_index": case_index,
                "task_id": task.task_id,
                "logical_domain_id": task.logical_domain_id,
                "source": source,
                "candidate_index": candidate_index,
                "normalized_parameters": [float(value) for value in continuous],
                "episode_seed": int(task.geometry_seed + 200_000 + case_index),
                "action": NormalizedScenarioAction(
                    candidate_index,
                    tuple(float(value) for value in continuous),
                ),
            })
    return cases


def _random_inner_policy(seed: int):
    generator = np.random.default_rng(seed)

    def policy(_state: np.ndarray) -> np.ndarray:
        return generator.uniform(-1.0, 1.0, size=4).astype(np.float32)

    return policy


def _paired_record(case: dict[str, Any], policy: str, episode: Any) -> dict[str, Any]:
    outcome = episode.outcome
    return {
        **{key: value for key, value in case.items() if key != "action"},
        "policy": policy,
        "valid": bool(outcome["is_valid_episode"]),
        "failure": bool(outcome["is_failure"]),
        "target_collision": bool(outcome["valid_target_collision"]),
        "critical_near_miss": bool(outcome["valid_critical_near_miss"]),
        "event": bool(
            outcome["valid_target_collision"]
            or outcome["valid_critical_near_miss"]
        ),
        "min_ttc": float(outcome["min_ttc"]),
        "min_distance": float(outcome["min_distance"]),
        "challenge_min_ttc": outcome["challenge_min_ttc"],
        "challenge_min_distance": outcome["challenge_min_distance"],
        "termination_reason": outcome["termination_reason"],
    }


def _paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        domains.setdefault(row["logical_domain_id"], []).append(row)

    def summarize_policy(values: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "cases": len(values),
            "valid_rate": _rate(values, "valid"),
            "event_rate": _rate(values, "event"),
            "target_collision_rate": _rate(values, "target_collision"),
            "critical_near_miss_rate": _rate(values, "critical_near_miss"),
            "failure_rate": _rate(values, "failure"),
        }

    report = {}
    for domain, values in sorted(domains.items()):
        sac = [row for row in values if row["policy"] == "sac"]
        random = [row for row in values if row["policy"] == "random"]
        sac_summary = summarize_policy(sac)
        random_summary = summarize_policy(random)
        report[domain] = {
            "sac": sac_summary,
            "random": random_summary,
            "event_rate_gain": sac_summary["event_rate"] - random_summary["event_rate"],
            "target_collision_rate_gain": (
                sac_summary["target_collision_rate"]
                - random_summary["target_collision_rate"]
            ),
            "valid_rate_gain": sac_summary["valid_rate"] - random_summary["valid_rate"],
        }
    return report


def run_paired(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Run the fixed 9-case in-domain SAC/Random engineering comparison."""
    config, taskbook, device = load_config(config_path)
    checkpoint = HierarchicalCheckpoint.load(
        checkpoint_path, expected_config_hash=checkpoint_config_hash(config),
    )
    assert_taskbook_compatible(checkpoint, taskbook)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    criteria = FailureCriteria.from_config(config["failure"])
    cases = paired_cases(config, taskbook)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_stage1_cases.json").write_text(
        json.dumps([
            {key: value for key, value in case.items() if key != "action"}
            for case in cases
        ], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tasks = {task.task_id: task for task in _training_tasks(config, taskbook)}
    records = []
    for case in cases:
        task = tasks[case["task_id"]]
        online = build_online(
            model, task, int(config["training"]["step_budget"]), criteria,
        )
        for policy in ("sac", "random"):
            episode = online.run(
                task,
                1,
                deterministic=True,
                posterior_support_limit=0,
                scene_action_provider=lambda *_args, value=case["action"]: value,
                inner_action_provider=(
                    None if policy == "sac"
                    else _random_inner_policy(20260906 + case["case_index"])
                ),
                episode_seed_provider=lambda *_args, value=case["episode_seed"]: value,
            ).episodes[0]
            records.append(_paired_record(case, policy, episode))
    report = {
        "scope": {
            "functional_scenario": "cutin",
            "sut_ref": "idm_normal",
            "geometry_id": "cutin-g01",
            "logical_domains": sorted({case["logical_domain_id"] for case in cases}),
            "cases_per_domain": 3,
            "outer_trained": False,
            "ood_split_accessed": False,
        },
        "checkpoint": str(checkpoint_path),
        "paired_protocol": (
            "SAC and Random share every concrete Logical action, candidate, episode seed, and horizon."
        ),
        "paired_gain_status": "inconclusive",
        "domain_summary": _paired_summary(records),
        "records": records,
    }
    (output / "paired_stage1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-output")
    args = parser.parse_args()
    report = run(args.config, args.checkpoint)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    if args.paired_output:
        run_paired(args.config, args.checkpoint, args.paired_output)


if __name__ == "__main__":
    main()
