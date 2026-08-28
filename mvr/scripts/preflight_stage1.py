"""Run the staged, budget-aware checks that precede Formal Stage1.

The configuration selects which simulator phases to run.  Every omitted phase
is recorded as skipped and therefore cannot accidentally produce a
``PRE_STAGE1_PASS`` decision.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..failure.criteria import FailureCriteria
from ..scenario.option import AdversarialOption
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.taskbook import load_taskbook
from ..training.pipeline import build_model, load_config
from ..training.trainers import train_inner
from ..training.stage1_sampling import PretrainSceneSampler
from ..validation.stage1_preflight import (
    FAMILIES,
    audit_event_bonus_once,
    audit_learning,
    audit_nuisance_invariance,
    audit_parameter_update,
    audit_preflight_gates,
    audit_profile_effect,
    audit_reachability,
    audit_replay_contract,
    audit_reward,
)
from .sweep_stage1_actions import run as run_reachability_sweep
from ..training.trainers import build_online


DEFAULT_PHASES = (
    "contracts",
    "reward",
    "reachability",
    "p1",
    "p4",
    "p5",
    "p6",
    "nuisance",
    "profile",
)
VIOLATION_FIELDS = (
    "non_target_collision",
    "adversary_out_of_road",
    "sut_out_of_road",
    "wrong_route",
    "adversary_traffic_violation",
)


def _task_map(taskbook: Path) -> dict[str, Any]:
    return {task.task_id: task for task in load_taskbook(taskbook)}


def _tasks(taskbook: Path, task_ids: Sequence[str]) -> list[Any]:
    available = _task_map(taskbook)
    missing = [task_id for task_id in task_ids if task_id not in available]
    if missing:
        raise ValueError(f"preflight taskbook is missing tasks: {', '.join(missing)}")
    return [available[task_id] for task_id in task_ids]


def _base_scene_action(*_: Any) -> NormalizedScenarioAction:
    return NormalizedScenarioAction(
        candidate_index=0,
        continuous=(0.0, 0.0, 0.0, 0.0, 0.0),
        option=AdversarialOption.APPROACH_CONFLICT,
    )


def _episode_record(task: Any, episode: Any) -> dict[str, Any]:
    outcome = dict(episode.rollout.outcome)
    return {
        "family": task.functional_scenario,
        "task_id": task.task_id,
        "valid": bool(outcome.get("is_valid_episode", False)),
        "failure": bool(outcome.get("is_failure", False)),
        "violations": {field: bool(outcome.get(field, False)) for field in VIOLATION_FIELDS},
        "event_kind": outcome.get("event_kind"),
        "min_ttc": float(outcome.get("min_ttc", 0.0)),
        "min_distance": float(outcome.get("min_distance", 0.0)),
        "termination_reason": outcome.get("termination_reason"),
        "semantic_challenge_steps": int(sum(
            bool(row["info"].get("semantic_challenge_phase_active", False))
            for row in episode.rollout.transitions
        )),
        "transitions": len(episode.rollout.transitions),
    }


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"episodes": 0, "valid_rate": 0.0, "violation_rate": 1.0}
        return {
            "episodes": len(rows),
            "valid_rate": float(np.mean([row["valid"] for row in rows])),
            "valid_critical_rate": float(np.mean([row["failure"] for row in rows])),
            "violation_rate": float(np.mean([
                any(row["violations"].values()) for row in rows
            ])),
            "median_min_ttc": float(np.median([row["min_ttc"] for row in rows])),
            "median_min_distance": float(np.median([row["min_distance"] for row in rows])),
            "valid_event_count": int(sum(row["failure"] for row in rows)),
            "challenge_phase_rate": float(sum(row["semantic_challenge_steps"] for row in rows) / max(
                sum(row["transitions"] for row in rows), 1
            )),
        }

    return {
        "overall": summary(records),
        "by_family": {
            family: summary([row for row in records if row["family"] == family])
            for family in FAMILIES
        },
        "rows": list(records),
    }


def _run_base_contract(
    config: Mapping[str, Any], taskbook: Path, device: torch.device, criteria: FailureCriteria
) -> dict[str, Any]:
    settings = config.get("preflight", {})
    task_ids = settings.get("p1_task_ids", ())
    tasks = _tasks(taskbook, task_ids)
    model = build_model(dict(config), device)
    model.eval()
    records = []
    for task in tasks:
        result = build_online(model, task, int(config["training"]["step_budget"]), criteria).run(
            task, 1, deterministic=True, posterior_support_limit=0,
            scene_action_provider=_base_scene_action,
            inner_action_provider=lambda _: np.zeros(2, dtype=np.float32),
        )
        records.extend(_episode_record(task, episode) for episode in result.episodes)
    report = _summarize_records(records)
    report["pass"] = bool(
        len(records) == len(tasks)
        and all(row["valid"] and not any(row["violations"].values()) for row in records)
    )
    return report


def _component_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _make_optimizer(model: Any, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    components = ("map_encoder", "interaction_encoder", "shared_feature_encoder", "option_embedding", "inner_sac")
    parameters = [
        parameter
        for name in components
        for parameter in model.training_components()[name].parameters()
        if parameter.requires_grad
    ]
    return torch.optim.Adam(parameters, lr=float(config["inner"].get("learning_rate", 3e-4)))


def _run_p4(
    config: Mapping[str, Any], taskbook: Path, device: torch.device, criteria: FailureCriteria
) -> dict[str, Any]:
    local = copy.deepcopy(dict(config))
    settings = dict(local.get("preflight", {}).get("p4", {}))
    local["inner"] = {**local["inner"], **settings}
    tasks = _tasks(taskbook, local.get("preflight", {}).get("p4_task_ids", ()))
    model = build_model(local, device)
    model.train()
    before = _component_snapshot(model.inner_sac)
    metrics, replay = train_inner(model, tasks, local, criteria, _make_optimizer(model, local))
    after = _component_snapshot(model.inner_sac)
    losses = [metrics["last_loss"]] if metrics.get("last_loss") else []
    replay_report = audit_replay_contract(replay.rows)
    update_report = audit_parameter_update(before, after, losses)
    signal = metrics.get("training_signal", {})
    signal_report = {
        "pass": all(f"family:{family}" in signal for family in FAMILIES),
        "buckets": sorted(signal),
    }
    saturation = float(metrics.get("action_saturation_rate", 1.0))
    return {
        "pass": bool(replay_report["pass"] and update_report["pass"] and signal_report["pass"] and saturation < 0.95),
        "episodes": len(tasks) * int(local["inner"]["episodes_per_task"]),
        "metrics": metrics,
        "replay": replay_report,
        "parameter_update": update_report,
        "training_signal": signal_report,
        "action_saturation_rate": saturation,
    }


def _policy_records(
    model: Any,
    tasks: Sequence[Any],
    config: Mapping[str, Any],
    criteria: FailureCriteria,
    provider: Callable[[np.ndarray], np.ndarray] | None,
    cases_per_task: int,
) -> list[dict[str, Any]]:
    sampler = PretrainSceneSampler(tuple(tasks), cases_per_task, int(config["seed"]) + 101)
    records = []
    for task in tasks:
        result = build_online(model, task, int(config["training"]["step_budget"]), criteria).run(
            task, cases_per_task, deterministic=True, posterior_support_limit=0,
            scene_action_provider=sampler, inner_action_provider=provider,
        )
        records.extend(_episode_record(task, episode) for episode in result.episodes)
    return records


def _run_p5(
    config: Mapping[str, Any], taskbook: Path, device: torch.device, criteria: FailureCriteria
) -> dict[str, Any]:
    local = copy.deepcopy(dict(config))
    local["inner"] = {**local["inner"], **dict(local.get("preflight", {}).get("p5", {}))}
    train_tasks = _tasks(taskbook, local.get("preflight", {}).get("p5_task_ids", ()))
    model = build_model(local, device)
    model.train()
    metrics, _ = train_inner(model, train_tasks, local, criteria, _make_optimizer(model, local))
    model.eval()
    validation_tasks = [
        task for task in load_taskbook(taskbook)
        if task.sut_split == "validation" and task.geometry_split == "validation" and task.functional_split == "train"
    ]
    cases = int(local.get("preflight", {}).get("p5_validation_cases", 2))
    random_rng = np.random.default_rng(int(local["seed"]) + 303)
    policies = {
        "base": _summarize_records(_policy_records(model, validation_tasks, local, criteria, lambda _: np.zeros(2, dtype=np.float32), cases)),
        "random_residual": _summarize_records(_policy_records(model, validation_tasks, local, criteria, lambda _: random_rng.uniform(-1.0, 1.0, 2).astype(np.float32), cases)),
        "trained_inner": _summarize_records(_policy_records(model, validation_tasks, local, criteria, None, cases)),
    }
    signal_report = metrics.get("training_signal", {})
    learning = audit_learning(policies, signal_report)
    return {"pass": learning["pass"], "training_metrics": metrics, "policies": policies, "learning": learning}


def _run_p6(
    config: Mapping[str, Any], taskbook: Path, device: torch.device, criteria: FailureCriteria
) -> dict[str, Any]:
    local = copy.deepcopy(dict(config))
    local["inner"] = {**local["inner"], **dict(local.get("preflight", {}).get("p6", {}))}
    tasks = [
        task for task in load_taskbook(taskbook)
        if task.sut_split == "train" and task.geometry_split == "train" and task.functional_split == "train"
    ]
    model = build_model(local, device)
    model.train()
    metrics, _ = train_inner(model, tasks, local, criteria, _make_optimizer(model, local))
    clone = build_model(local, device)
    clone.load_state_dict(model.state_dict())
    counts = metrics.get("task_episode_counts", {})
    coverage = {
        "tasks": len(tasks),
        "all_tasks_once": len(counts) == len(tasks) and all(int(value) == 1 for value in counts.values()),
        "families": sorted(metrics.get("family_episode_counts", {})),
        "geometries": len(metrics.get("geometry_episode_counts", {})),
        "sut_profiles": len(metrics.get("sut_episode_counts", {})),
        "candidates": len(metrics.get("candidate_episode_counts", {})),
        "options": len(metrics.get("option_episode_counts", {})),
        "checkpoint_reloadable": all(torch.isfinite(value).all().item() for value in clone.state_dict().values()),
        "training_signal_buckets": sorted(metrics.get("training_signal", {})),
    }
    coverage["pass"] = bool(
        coverage["tasks"] == 36 and coverage["all_tasks_once"]
        and coverage["families"] == sorted(FAMILIES)
        and coverage["geometries"] == 9 and coverage["sut_profiles"] == 4
        and coverage["candidates"] == 7 and coverage["options"] == 3
        and coverage["checkpoint_reloadable"]
    )
    return {"pass": coverage["pass"], "coverage": coverage, "metrics": metrics}


def _digest(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy().tobytes()
    elif isinstance(value, np.ndarray):
        value = value.tobytes()
    else:
        value = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _run_nuisance(
    config: Mapping[str, Any], taskbook: Path, device: torch.device, criteria: FailureCriteria
) -> dict[str, Any]:
    model = build_model(dict(config), device)
    model.eval()
    all_tasks = _task_map(taskbook)
    rows = []
    for family in ("merge", "roundabout"):
        task = next(task for task in all_tasks.values() if task.functional_scenario == family and task.sut_split == "validation" and task.geometry_split == "validation")
        sampler = PretrainSceneSampler((task,), 4, int(config["seed"]) + 707)
        for case_id in range(4):
            for onset in (0.2, 0.5, 0.8):
                def action_provider(task_arg: Any, episode_index: int, candidates: Sequence[Any], space: Any, onset_value: float = onset) -> NormalizedScenarioAction:
                    action = sampler(task_arg, episode_index, candidates, space)
                    controls = list(action.continuous)
                    controls[4] = onset_value
                    return NormalizedScenarioAction(action.candidate_index, tuple(controls), action.option)

                episode = build_online(model, task, int(config["training"]["step_budget"]), criteria).run(
                    task, 1, deterministic=True, posterior_support_limit=0,
                    episode_index_offset=case_id, scene_action_provider=action_provider,
                    inner_action_provider=lambda _: np.zeros(2, dtype=np.float32),
                ).episodes[0]
                outcome = episode.rollout.outcome
                rows.append({
                    "family": family,
                    "case_id": str(case_id),
                    "onset": onset,
                    "trajectory_digest": _digest(episode.rollout.trajectory),
                    "outcome_digest": _digest({key: outcome.get(key) for key in ("event_kind", "min_ttc", "min_distance", "termination_reason")}),
                })
    return audit_nuisance_invariance(rows)


def _run_profiles(
    config: Mapping[str, Any], taskbook: Path, device: torch.device, criteria: FailureCriteria
) -> dict[str, Any]:
    model = build_model(dict(config), device)
    model.eval()
    profiles = tuple(option.value for option in AdversarialOption)
    rows = []
    for family in FAMILIES:
        task = next(task for task in load_taskbook(taskbook) if task.functional_scenario == family and task.sut_split == "validation" and task.geometry_split == "validation")
        for profile in profiles:
            option = AdversarialOption(profile)
            def action_provider(*_: Any, selected_option: AdversarialOption = option) -> NormalizedScenarioAction:
                return NormalizedScenarioAction(0, (0.0, 0.0, 0.0, 0.0, 0.0), selected_option)

            episode = build_online(model, task, int(config["training"]["step_budget"]), criteria).run(
                task, 1, deterministic=True, posterior_support_limit=0,
                scene_action_provider=action_provider,
                inner_action_provider=lambda _: np.zeros(2, dtype=np.float32),
            ).episodes[0]
            outcome = episode.rollout.outcome
            rows.append({"family": family, "profile": profile, "min_ttc": outcome.get("min_ttc"), "min_distance": outcome.get("min_distance")})
    return audit_profile_effect(rows, profiles)


def _run_contracts(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    model = build_model(dict(config), device)
    checks = {
        "state_dim": model.state_dim == 11,
        "action_dim": int(config["inner"].get("action_dim", 0)) == 2,
        "action_schema": config["control"].get("action_schema") == "vehicle_residual_2d",
        "nominal_controller_schema": config["control"].get("nominal_controller_schema") == "metadrive_lane_stable_idm",
        "scenario_contract_schema": config["control"].get("scenario_contract_schema") == "scenario_contract",
    }
    commands = {}
    root = Path.cwd()
    for name, command in {
        "pytest": [sys.executable, "-m", "pytest", "mvr/tests", "-q"],
        "compileall": [sys.executable, "-m", "compileall", "-q", "mvr"],
    }.items():
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        commands[name] = {"returncode": completed.returncode, "passed": completed.returncode == 0, "tail": completed.stdout[-1000:]}
    checks.update({name: row["passed"] for name, row in commands.items()})
    return {"pass": bool(all(checks.values())), "checks": checks, "commands": commands}


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    phases: Sequence[str] | None = None,
) -> dict[str, Any]:
    config, taskbook, device = load_config(config_path)
    preflight = dict(config.get("preflight", {}))
    requested = tuple(phases if phases is not None else preflight.get("phases", DEFAULT_PHASES))
    criteria = FailureCriteria.from_config(config["failure"])
    reports: dict[str, dict[str, Any]] = {}
    if "contracts" in requested:
        reports["P0_engineering"] = _run_contracts(config, device)
    if "reward" in requested:
        reports["P3_reward"] = {
            "curve": audit_reward(criteria, objective=str(preflight.get("reward_objective", "threshold_proximity"))),
            "event_bonus": audit_event_bonus_once(criteria),
        }
        reports["P3_reward"]["pass"] = bool(
            reports["P3_reward"]["curve"]["pass"] and reports["P3_reward"]["event_bonus"]["pass"]
        )
    if "reachability" in requested:
        sweep_path = Path(preflight.get("reachability_report", "results/mvr/diagnostics/s0_action_reachability.json"))
        sweep = run_reachability_sweep(config_path, sweep_path)
        reports["P2_reachability"] = audit_reachability(sweep["summary"])
        reports["P2_reachability"]["report"] = str(sweep_path)
    if "p1" in requested:
        reports["P1_scenario"] = _run_base_contract(config, taskbook, device, criteria)
    if "p4" in requested:
        reports["P4_sac_plumbing"] = _run_p4(config, taskbook, device, criteria)
    if "p5" in requested:
        reports["P5_shared_learning"] = _run_p5(config, taskbook, device, criteria)
    if "p6" in requested:
        reports["P6_coverage"] = _run_p6(config, taskbook, device, criteria)
    if "nuisance" in requested:
        reports["nuisance_invariance"] = _run_nuisance(config, taskbook, device, criteria)
    if "profile" in requested:
        reports["profile_effect"] = _run_profiles(config, taskbook, device, criteria)
    gates = {
        name: reports.get(name, {"pass": False, "status": "skipped"})
        for name in (
            "P0_engineering", "P1_scenario", "P2_reachability", "P3_reward",
            "P4_sac_plumbing", "P5_shared_learning", "P6_coverage",
            "nuisance_invariance", "profile_effect",
        )
    }
    decision = audit_preflight_gates(gates)
    report = {
        "stage": "pre_formal_stage1",
        "requested_phases": list(requested),
        "device": str(device),
        "reports": reports,
        "decision": {"pre_stage1_pass": decision["pass"], **decision},
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage1 preflight checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    if not report["decision"]["pre_stage1_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
