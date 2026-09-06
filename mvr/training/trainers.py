"""Stage-local optimization loops for the canonical MVR pipeline."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import torch

from ..failure.criteria import FailureCriteria
from ..model import TransferableScenarioMiner
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.task_spec import ScenarioMiningTaskSpec
from .online_meta_test import OnlineMetaTest
from .meta_sampler import MetaTaskSampler
from .replay import ContextReplay, InnerReplay, SupportGroup
from .runner import HierarchicalRunner
from .stage1_sampling import PretrainSceneSampler
from .updates import update_inner_sac, update_outer_ppo


def build_online(
    model: TransferableScenarioMiner,
    _task: ScenarioMiningTaskSpec,
    max_steps: int,
    criteria: FailureCriteria,
    executor: ScenarioExecutor | None = None,
) -> OnlineMetaTest:
    return OnlineMetaTest(
        model,
        executor or ScenarioExecutor(load_adapters(), mvr_parameter_spaces()),
        HierarchicalRunner(max_steps, criteria),
    )


def _replay(settings: dict[str, Any], rows: list[Any] | None) -> InnerReplay:
    return InnerReplay(capacity=int(settings.get("replay_capacity", 100_000)), rows=list(rows or ()))


def _update_inner(
    model: TransferableScenarioMiner,
    replay: InnerReplay,
    optimizer: torch.optim.Optimizer,
    settings: dict[str, Any],
    losses: list[dict[str, float]],
    context_replay: ContextReplay | None = None,
) -> None:
    batch_size = int(settings.get("batch_size", 64))
    for _ in range(int(settings["updates_per_episode"])):
        if len(replay.rows) >= batch_size:
            losses.append(
                update_inner_sac(
                    model,
                    replay,
                    optimizer,
                    batch_size=batch_size,
                    gradient_clip_norm=float(settings.get("gradient_clip_norm", 5.0)),
                    event_sample_fraction=float(settings.get("event_sample_fraction", 0.25)),
                    event_action_weight=float(settings.get("event_action_weight", 0.0)),
                    gamma=float(settings.get("gamma", 0.99)),
                    context_replay=context_replay,
                )
            )


def train_interaction_prior(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
    scene_action_provider: Callable[
        [ScenarioMiningTaskSpec, int, tuple[Any, ...], Any], Any
    ] | None = None,
) -> tuple[dict[str, Any], InnerReplay]:
    settings = dict(config["interaction_prior"])
    episodes_per_task = int(settings["episodes_per_task"])
    if episodes_per_task < 1:
        raise ValueError("inner episodes_per_task must be positive")
    max_steps = int(config["training"]["step_budget"])
    replay, losses = _replay(settings, None), []
    sampler = MetaTaskSampler(tasks)
    cutin_inner = config.get("cutin_inner", {})
    accident_search = cutin_inner.get("accident_search", {})
    scene_sampler = scene_action_provider or PretrainSceneSampler(
        tuple(tasks), episodes_per_task, int(config["seed"]),
        eta_offsets_s=accident_search.get("eta_offsets_s"),
    )
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    online = build_online(model, tasks[0], max_steps, criteria, executor)
    episodes = []
    episode_optimizer_updates: list[int] = []
    transitions_collected = 0
    warmup_episodes = int(settings.get("warmup_episodes", 0))
    for episode_index in range(episodes_per_task):
        for task in sampler.shuffled_epoch():
            inner_action_provider = None
            if episode_index < warmup_episodes:
                rng = np.random.default_rng(
                    int(config["seed"]) + 1000 * episode_index
                )

                def random_inner_action(
                    _state: np.ndarray,
                    generator: np.random.Generator = rng,
                ) -> np.ndarray:
                    return generator.uniform(-1.0, 1.0, size=(4,)).astype(
                        np.float32
                    )

                inner_action_provider = random_inner_action
            result = online.run(
                task,
                1,
                posterior_support_limit=0,
                episode_index_offset=episode_index,
                scene_action_provider=scene_sampler,
                inner_action_provider=inner_action_provider,
                inner_gamma=float(settings.get("gamma", 0.99)),
            )
            print(
                f"inner episode {len(episodes) + 1}/{len(tasks) * episodes_per_task}: "
                f"task={task.task_id}",
                flush=True,
            )
            for row in result.inner_transitions:
                replay.add(row)
            transitions_collected += len(result.inner_transitions)
            episodes.extend((task, episode) for episode in result.episodes)
            updates_before = len(losses)
            if episode_index >= warmup_episodes:
                _update_inner(model, replay, optimizer, settings, losses)
            episode_optimizer_updates.extend(
                [len(losses) - updates_before] * len(result.episodes)
            )
    metrics = _inner_metrics(
        len(episodes), replay, losses, episodes, transitions_collected,
        gamma=float(settings.get("gamma", 0.99)),
    )
    for record, updates in zip(
        metrics["reward_episode_records"], episode_optimizer_updates
    ):
        record.update({
            "run_id": "interaction_prior",
            "action_schema": config["control"]["action_schema"],
            "reward_schema": "inner_risk_reward_components_v1",
            "training_seed": int(config["seed"]),
            "warmup": bool(record["domain_episode_index"] <= warmup_episodes),
            "optimizer_updates": updates,
        })
    metrics.update({
        "balanced_sampling_epochs": episodes_per_task,
        "updates_per_episode": int(settings["updates_per_episode"]),
        "requested_optimizer_updates": len(episodes) * int(settings["updates_per_episode"]),
        "warmup_skipped_updates": len(episodes) * int(settings["updates_per_episode"]) - len(losses),
    })
    return metrics, replay


def train_context_meta(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], InnerReplay]:
    """Train query transitions from task-local, disjoint support groups."""
    settings = dict(config["context_meta"])
    support_choices = tuple(int(value) for value in settings["support_shots"])
    groups_per_task = int(settings["groups_per_task"])
    queries_per_group = int(settings["queries_per_group"])
    if not support_choices or min(support_choices) < 1 or groups_per_task < 1 or queries_per_group < 1:
        raise ValueError("context meta-training requires positive support and query budgets")
    max_episodes = groups_per_task * (max(support_choices) + queries_per_group)
    cutin_inner = config.get("cutin_inner", {})
    accident_search = cutin_inner.get("accident_search", {})
    sampler = PretrainSceneSampler(
        tuple(tasks), max_episodes, int(config["seed"]),
        eta_offsets_s=accident_search.get("eta_offsets_s"),
    )
    replay, context, losses, episodes = _replay(settings, None), ContextReplay(), [], []
    online = build_online(model, tasks[0], int(config["training"]["step_budget"]), criteria)
    for task_index, task in enumerate(MetaTaskSampler(tasks).shuffled_epoch()):
        for group_index in range(groups_per_task):
            support_count = support_choices[(task_index + group_index) % len(support_choices)]
            offset = group_index * (max(support_choices) + queries_per_group)
            result = online.run(
                task, support_count + queries_per_group,
                posterior_support_limit=support_count, episode_index_offset=offset,
                scene_action_provider=sampler,
                inner_gamma=float(settings.get("gamma", 0.99)),
            )
            support, query = result.episodes[:support_count], result.episodes[support_count:]
            group_id = f"{task.task_id}:support:{group_index}"
            group = SupportGroup(group_id, task.task_id, tuple(support), {
                episode.episode_id: episode for episode in query
            })
            context.add(group)
            query_ids = set(group.query_episodes)
            for row in result.inner_transitions:
                if row.episode_id in query_ids:
                    replay.add(replace(row, support_group_id=group_id))
            episodes.extend((task, episode) for episode in result.episodes)
            _update_inner(model, replay, optimizer, settings, losses, context)
    metrics = _inner_metrics(len(episodes), replay, losses, episodes)
    metrics.update({
        "support_groups": len(context.groups),
        "support_shots": list(support_choices),
        "queries_per_group": queries_per_group,
    })
    return metrics, replay


def _inner_metrics(
    episodes: int,
    replay: InnerReplay,
    losses: list[dict[str, float]],
    episode_records: list[tuple[ScenarioMiningTaskSpec, Any]] | None = None,
    transitions_collected: int | None = None,
    gamma: float = 0.99,
) -> dict[str, Any]:
    records = list(episode_records or ())
    task_counts = {task.task_id: 0 for task, _ in records}
    family_counts: dict[str, int] = {}
    geometry_counts: dict[str, int] = {}
    sut_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    returns, valid, failures, min_ttc, min_distance, closing = [], [], [], [], [], []
    action_values, saturated = [], []
    for task, episode in records:
        task_counts[task.task_id] += 1
        family_counts[task.functional_scenario] = family_counts.get(task.functional_scenario, 0) + 1
        geometry_counts[task.geometry_id] = geometry_counts.get(task.geometry_id, 0) + 1
        sut_counts[task.sut_ref] = sut_counts.get(task.sut_ref, 0) + 1
        scenario = episode.concrete_scenario
        candidate_counts[scenario.candidate_id] = candidate_counts.get(scenario.candidate_id, 0) + 1
        returns.append(sum(float(row["reward_inner"]) for row in episode.rollout.transitions))
        valid.append(float(episode.rollout.signature.is_valid_episode))
        failures.append(float(episode.rollout.signature.is_failure))
        outcome = episode.rollout.outcome
        min_ttc.append(float(outcome.get("min_ttc", 0.0)))
        min_distance.append(float(outcome.get("min_distance", 0.0)))
        closing.append(float(outcome.get("max_closing_speed", 0.0)))
        for row in episode.rollout.transitions:
            action = np.asarray(row["planner_action"], dtype=np.float32)
            action_values.extend(np.abs(action).tolist())
            saturated.extend((np.abs(action) > 0.95).tolist())

    training_signal = _training_signal_metrics(records)
    episode_return_curve = [
        {
            "episode": index + 1,
            "task_id": task.task_id,
            "sut_ref": task.sut_ref,
            "geometry_id": task.geometry_id,
            "logical_domain_id": task.logical_domain_id,
            "inner_return": float(sum(
                float(row["reward_inner"]) for row in episode.rollout.transitions
            )),
            "valid": bool(episode.rollout.signature.is_valid_episode),
            "failure": bool(episode.rollout.signature.is_failure),
        }
        for index, (task, episode) in enumerate(records)
    ]

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    actor = [value["inner_actor_loss"] for value in losses if "inner_actor_loss" in value]
    critic = [value["inner_critic_loss"] for value in losses if "inner_critic_loss" in value]
    alpha = [value["inner_alpha_loss"] for value in losses if "inner_alpha_loss" in value]
    td_target = [value["inner_td_target_variance"] for value in losses if "inner_td_target_variance" in value]
    reward_episode_records = _reward_episode_records(records, gamma)
    metrics = {
        "simulator_episodes_consumed": episodes,
        "transitions": transitions_collected if transitions_collected is not None else len(replay.rows),
        "replay_transitions": len(replay.rows),
        "optimizer_updates": len(losses),
        "last_loss": losses[-1] if losses else {},
        "task_episode_counts": task_counts,
        "family_episode_counts": family_counts,
        "geometry_episode_counts": geometry_counts,
        "sut_episode_counts": sut_counts,
        "candidate_episode_counts": candidate_counts,
        "mean_inner_episode_return": mean(returns),
        "mean_valid_rate": mean(valid),
        "mean_failure_rate": mean(failures),
        "mean_min_ttc": mean(min_ttc),
        "mean_min_distance": mean(min_distance),
        "mean_max_closing_speed": mean(closing),
        "actor_loss_mean": mean(actor),
        "actor_loss_last": actor[-1] if actor else None,
        "critic_loss_mean": mean(critic),
        "critic_loss_last": critic[-1] if critic else None,
        "alpha_loss_mean": mean(alpha),
        "alpha_loss_last": alpha[-1] if alpha else None,
        "td_target_variance_mean": mean(td_target),
        "td_target_variance_last": td_target[-1] if td_target else None,
        "action_abs_mean": mean(action_values),
        "action_saturation_rate": mean([float(value) for value in saturated]),
        "training_signal": training_signal,
        "episode_return_curve": episode_return_curve,
        "reward_episode_records": reward_episode_records,
        "reward_component_summary": _reward_component_summary(
            reward_episode_records
        ),
    }
    return metrics


def _reward_episode_records(
    records: list[tuple[ScenarioMiningTaskSpec, Any]],
    gamma: float,
) -> list[dict[str, Any]]:
    """Build the compact, per-episode reward and control audit records."""
    domain_indices: dict[str, int] = {}
    component_names = (
        "criticality_previous",
        "criticality_current",
        "reward_risk",
        "reward_event",
        "reward_progress",
        "penalty_tracking",
        "penalty_shield",
        "penalty_invalid",
        "reward_preclip",
        "reward_clip_adjustment",
        "reward_total",
    )
    output = []
    for task, episode in records:
        domain = task.logical_domain_id
        domain_indices[domain] = domain_indices.get(domain, 0) + 1
        transitions = episode.rollout.transitions
        components = {
            name: float(sum(
                float(row["reward_components"][name]) for row in transitions
            ))
            for name in component_names
        }
        outcome = episode.rollout.outcome
        telemetry = outcome.get("control_telemetry", {})
        raw = np.asarray(telemetry.get("raw_longitudinal", ()), dtype=float)
        limited = np.asarray(
            telemetry.get("speed_limited_longitudinal", ()), dtype=float
        )
        projected = np.asarray(
            telemetry.get("projected_longitudinal", ()), dtype=float
        )
        scales = np.asarray(telemetry.get("path_projection_scale", ()), dtype=float)
        feasible = telemetry.get("path_speed_feasible", ())
        onset = outcome.get("cutin_actual_onset")
        output.append({
            "task_id": task.task_id,
            "logical_domain_id": domain,
            "domain_episode_index": domain_indices[domain],
            "episode_seed": episode.concrete_scenario.episode_seed,
            "candidate_index": episode.candidate_index,
            "normalized_initial_parameters": list(episode.continuous),
            "environment_steps": len(transitions),
            "macro_transitions": len(episode.macro_records),
            "inner_return_undiscounted": float(sum(
                float(row["reward_inner"]) for row in transitions
            )),
            "inner_return_discounted": float(sum(
                gamma ** index * float(row["reward_inner"])
                for index, row in enumerate(transitions)
            )),
            "reward_component_sums": components,
            "reward_clipped_steps": int(sum(
                abs(float(row["reward_components"]["reward_clip_adjustment"]))
                > 1e-12 for row in transitions
            )),
            "valid": bool(outcome["is_valid_episode"]),
            "valid_target_collision": bool(outcome["valid_target_collision"]),
            "valid_critical_near_miss": bool(
                outcome["valid_critical_near_miss"]
            ),
            "challenge_steps": int(sum(
                bool(row["info"]["semantic_challenge_phase_active"])
                for row in transitions
            )),
            "min_challenge_ttc": outcome["challenge_min_ttc"],
            "min_challenge_distance": outcome["challenge_min_distance"],
            "actual_onset_time": None if onset is None else onset["time_s"],
            "actual_onset_gap": None if onset is None else onset["gap_m"],
            "actual_onset_speeds": None if onset is None else {
                "adversary_speed_mps": onset["adversary_speed_mps"],
                "sut_speed_mps": onset["sut_speed_mps"],
            },
            "path_speed_infeasible_steps": int(sum(not bool(value) for value in feasible)),
            "path_projection_statistics": {
                "count": int(scales.size),
                "mean": None if not scales.size else float(scales.mean()),
                "minimum": None if not scales.size else float(scales.min()),
                "intervened_steps": int(sum(scales < 1.0 - 1e-6)),
            },
            "longitudinal_override_statistics": {
                "speed_limit_intervened_steps": int(sum(
                    np.abs(raw - limited) > 1e-6
                )),
                "projector_intervened_steps": int(sum(
                    np.abs(limited - projected) > 1e-6
                )),
                "speed_limit_abs_delta_mean": (
                    None if not raw.size else float(np.abs(raw - limited).mean())
                ),
                "projector_abs_delta_mean": (
                    None if not limited.size else float(
                        np.abs(limited - projected).mean()
                    )
                ),
            },
            "termination_reason": outcome["termination_reason"],
            "macro_records": list(episode.macro_records),
        })
    return output


def _reward_component_summary(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate the recorded component sums by Logical Domain."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["logical_domain_id"], []).append(record)
    return {
        domain: {
            name: float(np.mean([
                row["reward_component_sums"][name] for row in domain_records
            ]))
            for name in domain_records[0]["reward_component_sums"]
        }
        for domain, domain_records in sorted(grouped.items())
    }


def _training_signal_metrics(
    records: list[tuple[ScenarioMiningTaskSpec, Any]],
) -> dict[str, Any]:
    """Expose reward variation, challenge coverage, and event density."""
    buckets: dict[str, dict[str, Any]] = {}

    def bucket(name: str) -> dict[str, Any]:
        return buckets.setdefault(name, {
            "episodes": 0,
            "valid_event_episodes": 0,
            "valid_target_collision_episodes": 0,
            "valid_near_miss_episodes": 0,
            "transitions": 0,
            "positive_reward_transitions": 0,
            "event_capture_transitions": 0,
            "challenge_transitions": 0,
            "reward_values": [],
        })

    def add(name: str, episode: Any) -> None:
        values = bucket(name)
        outcome = episode.rollout.outcome
        values["episodes"] += 1
        values["valid_target_collision_episodes"] += int(
            bool(outcome.get("valid_target_collision", False))
        )
        values["valid_near_miss_episodes"] += int(
            bool(outcome.get("valid_critical_near_miss", False))
        )
        values["valid_event_episodes"] += int(
            bool(outcome.get("valid_target_collision", False))
            or bool(outcome.get("valid_critical_near_miss", False))
        )
        for row in episode.rollout.transitions:
            values["transitions"] += 1
            reward = float(row["reward_inner"])
            values["reward_values"].append(reward)
            values["positive_reward_transitions"] += int(reward > 0.0)
            values["event_capture_transitions"] += int(
                bool(row["info"].get("event_just_captured", False))
            )
            values["challenge_transitions"] += int(
                bool(row["info"].get("semantic_challenge_phase_active", False))
            )

    for task, episode in records:
        add("overall", episode)
        add(f"family:{task.functional_scenario}", episode)
        add(f"logical_domain:{task.logical_domain_id}", episode)

    report = {}
    for name, values in sorted(buckets.items()):
        rewards = np.asarray(values.pop("reward_values"), dtype=float)
        report[name] = {
            **values,
            "positive_reward_transition_fraction": float(
                values["positive_reward_transitions"] / max(values["transitions"], 1)
            ),
            "challenge_transition_fraction": float(
                values["challenge_transitions"] / max(values["transitions"], 1)
            ),
            "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "reward_variance": float(rewards.var()) if rewards.size else 0.0,
            "reward_range": float(rewards.max() - rewards.min()) if rewards.size else 0.0,
        }
    return report


def train_outer(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    settings = config["outer"]
    episodes_per_task = int(settings["episodes_per_task"])
    losses, episodes = [], 0
    for task in tasks:
        result = build_online(
            model, task, int(config["training"]["step_budget"]), criteria
        ).run(task, episodes_per_task)
        losses.append(
            update_outer_ppo(
                model.universal_scene_policy,
                result.outer_rollout,
                optimizer,
                epochs=int(settings["ppo_epochs"]),
                batch_size=int(settings["batch_size"]),
                router_balance_weight=float(settings.get("router_balance_weight", 0.01)),
            )
        )
        episodes += len(result.episodes)
    return {
        "simulator_episodes_consumed": episodes,
        "tasks": len(tasks),
        "episodes_per_task": episodes_per_task,
        "optimizer_updates": len(losses),
        "outer_ppo_loss": float(np.mean(losses)),
    }
