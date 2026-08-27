"""Validation-only end-to-end framework-pilot evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..failure.criteria import FailureCriteria
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
    seed_everything,
)
from ..training.stages import TrainingStage
from ..training.trainers import build_online
from .budget_protocol import BudgetProtocol
from .regimes import PILOT_VALIDATION_REGIME, select_regime_tasks


def _finite(values: Any) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def _tensor_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _episode_record(model: Any, episode: Any, outer_row: Any) -> dict[str, Any]:
    policy = model.universal_scene_policy
    scene = outer_row.scene_embedding.unsqueeze(0).to(model.device)
    latent = episode.latent_before.to(model.device)
    with torch.no_grad():
        router_logits = policy.router(scene, latent)
        router_weights = torch.softmax(router_logits, dim=-1)
    return {
        "episode_id": episode.episode_id,
        "episode_seed": episode.concrete_scenario.episode_seed,
        "geometry_hash": episode.concrete_scenario.geometry_hash,
        "candidate": episode.concrete_scenario.candidate_id,
        "x0": dict(episode.concrete_scenario.initial_state),
        "option": episode.concrete_scenario.option,
        "inner_policy_hash": episode.concrete_scenario.inner_policy_hash,
        "latent_before": _tensor_list(episode.latent_before),
        "latent_after": _tensor_list(episode.latent_after),
        "outer": {
            "expert_index": int(outer_row.expert_index.item()),
            "candidate_index": int(outer_row.candidate.item()),
            "continuous": _tensor_list(outer_row.continuous),
            "option_index": int(outer_row.option.item()),
            "router_weights": _tensor_list(router_weights),
        },
        "semantic_event": {
            key: episode.outcome.get(key)
            for key in (
                "event_kind", "event_semantic_valid", "event_traffic_valid",
                "valid_target_collision", "valid_critical_near_miss",
            )
        },
        "traffic_valid": bool(episode.rollout.signature.is_valid_episode),
        "failure_signature": {
            **asdict(episode.rollout.signature),
            "signature_id": episode.rollout.signature.signature_id,
        },
        "termination_reason": episode.outcome.get("termination_reason"),
        "sut_arrived_destination": bool(episode.outcome.get("sut_arrived_destination", False)),
    }


def _inner_latent_effect(model: Any, episode: Any) -> float:
    if not episode.rollout.transitions:
        return 0.0
    row = episode.rollout.transitions[0]
    state = torch.as_tensor(row["state"], dtype=torch.float32, device=model.device).unsqueeze(0)
    scene = episode.scene_embedding.to(model.device).unsqueeze(0)
    option = episode.option_index.to(model.device).reshape(1)
    config = episode.config.to(model.device).unsqueeze(0)
    with torch.no_grad():
        before = model.act_inner(state, scene, episode.latent_before.to(model.device), option, config, deterministic=True)
        after = model.act_inner(state, scene, episode.latent_after.to(model.device), option, config, deterministic=True)
    return float(torch.max(torch.abs(after - before)).item())


def _report_run(model: Any, online: Any, task: Any, budget: int, shot: int) -> dict[str, Any]:
    result = online.run(
        task,
        budget,
        deterministic=True,
        posterior_support_limit=shot,
    )
    records = [_episode_record(model, episode, outer_row) for episode, outer_row in zip(result.episodes, result.outer_rollout.rows)]
    latent_changes = [
        float(torch.linalg.vector_norm(episode.latent_after - episode.latent_before).item())
        for episode in result.episodes
    ]
    inner_effects = [_inner_latent_effect(model, episode) for episode in result.episodes]
    finite = all(
        _finite(record["latent_before"])
        and _finite(record["latent_after"])
        and _finite(record["outer"]["continuous"])
        and _finite(record["outer"]["router_weights"])
        for record in records
    )
    return {
        "task_id": task.task_id,
        "functional_scenario": task.functional_scenario,
        "support_shots": shot,
        "episode_budget": budget,
        "episodes": records,
        "posterior_latent_change_l2": latent_changes,
        "inner_action_change_linf": inner_effects,
        "finite": finite,
    }


def evaluate_pilot(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the validation-only MVR framework pilot.")
    parser.add_argument("--config", default="mvr/configs/mvr_pilot.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config, taskbook, device = load_config(args.config)
    checkpoint = HierarchicalCheckpoint.load(args.checkpoint, expected_config_hash=checkpoint_config_hash(config))
    if checkpoint.stage != TrainingStage.OUTER.value:
        raise ValueError("pilot evaluation requires an outer checkpoint")
    assert_taskbook_compatible(checkpoint, taskbook)
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    settings = config["evaluation"]
    protocol = BudgetProtocol(
        int(settings["total_episode_budget"]),
        tuple(int(value) for value in settings["support_shots"]),
    )
    protocol.validate()
    tasks = select_regime_tasks(load_taskbook(taskbook), PILOT_VALIDATION_REGIME, str(config["training"].get("family_filter", "all")))
    criteria = FailureCriteria.from_config(config["failure"])
    runs = []
    for seed in settings["seeds"]:
        seed_everything(int(seed))
        for task in tasks:
            online = build_online(model, task, int(config["training"]["step_budget"]), criteria)
            for shot in protocol.support_shots:
                runs.append(_report_run(model, online, task, protocol.total_episodes, shot))
    k0 = [change for run in runs if run["support_shots"] == 0 for change in run["posterior_latent_change_l2"]]
    k1 = [change for run in runs if run["support_shots"] == 1 for change in run["posterior_latent_change_l2"]]
    report = {
        "mode": "framework_pilot_validation_only",
        "regime": PILOT_VALIDATION_REGIME.name,
        "checkpoint_stage": checkpoint.stage,
        "device": str(device),
        "episode_accounting": {
            "tasks": len(tasks), "support_shots": list(protocol.support_shots),
            "episodes_per_run": protocol.total_episodes,
            "total_simulator_episodes": len(runs) * protocol.total_episodes,
        },
        "runs": runs,
        "checks": {
            "all_finite": all(run["finite"] for run in runs),
            "k0_stays_prior": bool(all(change <= 1e-7 for change in k0)),
            "k1_updates_posterior": bool(any(change > 1e-7 for change in k1)),
            "inner_responds_to_posterior": bool(any(
                effect > 1e-7 for run in runs if run["support_shots"] == 1
                for effect in run["inner_action_change_linf"]
            )),
            "moe_active": bool(all(
                _finite(episode["outer"]["router_weights"])
                and len(episode["outer"]["router_weights"]) == int(config["model"]["num_experts"])
                for run in runs for episode in run["episodes"]
            )),
            "posterior_changes_outer_proposal": False,
        },
    }
    # Compare the same scene under the K=0 prior and the adapted K=1 latent.
    for run in runs:
        if run["support_shots"] != 1 or len(run["episodes"]) < 2:
            continue
        first, second = run["episodes"][0], run["episodes"][1]
        if first["outer"] != second["outer"] or first["candidate"] != second["candidate"] or first["x0"] != second["x0"]:
            report["checks"]["posterior_changes_outer_proposal"] = True
            break
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    evaluate_pilot()
