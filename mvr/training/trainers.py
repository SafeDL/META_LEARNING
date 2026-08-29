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
    scene_sampler = scene_action_provider or PretrainSceneSampler(
        tuple(tasks), episodes_per_task, int(config["seed"])
    )
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    online = build_online(model, tasks[0], max_steps, criteria, executor)
    episodes = []
    transitions_collected = 0
    for episode_index in range(episodes_per_task):
        for task in sampler.shuffled_epoch():
            result = online.run(
                task,
                1,
                posterior_support_limit=0,
                episode_index_offset=episode_index,
                scene_action_provider=scene_sampler,
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
            _update_inner(model, replay, optimizer, settings, losses)
    metrics = _inner_metrics(
        len(episodes), replay, losses, episodes, transitions_collected,
    )
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
    sampler = PretrainSceneSampler(tuple(tasks), max_episodes, int(config["seed"]))
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
            action = np.asarray(row.get("executed_action", row["action"]), dtype=np.float32)
            action_values.extend(np.abs(action).tolist())
            saturated.extend((np.abs(action) > 0.95).tolist())

    training_signal = _training_signal_metrics(records)

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    actor = [value["inner_actor_loss"] for value in losses if "inner_actor_loss" in value]
    critic = [value["inner_critic_loss"] for value in losses if "inner_critic_loss" in value]
    alpha = [value["inner_alpha_loss"] for value in losses if "inner_alpha_loss" in value]
    td_target = [value["inner_td_target_variance"] for value in losses if "inner_td_target_variance" in value]
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
    }
    return metrics


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
