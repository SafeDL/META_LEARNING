"""Stage-local optimization loops for the canonical MVR pipeline."""
from __future__ import annotations

import random
from typing import Any

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
from .posterior_data import posterior_batch_from_episodes
from .replay import InnerReplay
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
                    event_action_weight=float(settings.get("event_action_weight", 2.0)),
                )
            )


def train_inner(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], InnerReplay]:
    settings = dict(config["inner"])
    episodes_per_task = int(settings["episodes_per_task"])
    if episodes_per_task < 1:
        raise ValueError("inner episodes_per_task must be positive")
    max_steps = int(config["training"]["step_budget"])
    replay, losses = _replay(settings, None), []
    sampler = MetaTaskSampler(tasks)
    scene_sampler = PretrainSceneSampler(tuple(tasks), episodes_per_task, int(config["seed"]))
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


def train_inner_latent_calibration(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], InnerReplay]:
    settings = dict(config["inner_latent_calibration"])
    support_count = int(settings["support_episodes"])
    if support_count < 1:
        raise ValueError("z-conditioned Inner calibration requires at least one support episode")
    budget = int(settings["simulator_episode_budget"])
    group_size = support_count + 1
    if budget % group_size:
        raise ValueError(
            "inner_latent_calibration simulator_episode_budget must divide support plus target"
        )
    replay, losses = _replay(settings, None), []
    sampler = MetaTaskSampler(tasks)
    for _ in range(budget // group_size):
        task = sampler.sample()
        result = build_online(model, task, int(config["training"]["step_budget"]), criteria).run(
            task, group_size, posterior_support_limit=support_count
        )
        for transition in result.inner_transitions:
            if transition.episode_id.endswith(f":{support_count}"):
                replay.add(transition)
        _update_inner(model, replay, optimizer, settings, losses)
    metrics = _inner_metrics(budget, replay, losses)
    metrics["calibration_groups"] = budget // group_size
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
    option_counts: dict[str, int] = {}
    returns, valid, failures, min_ttc, min_distance, closing = [], [], [], [], [], []
    action_values, saturated = [], []
    for task, episode in records:
        task_counts[task.task_id] += 1
        family_counts[task.functional_scenario] = family_counts.get(task.functional_scenario, 0) + 1
        geometry_counts[task.geometry_id] = geometry_counts.get(task.geometry_id, 0) + 1
        sut_counts[task.sut_ref] = sut_counts.get(task.sut_ref, 0) + 1
        scenario = episode.concrete_scenario
        candidate_counts[scenario.candidate_id] = candidate_counts.get(scenario.candidate_id, 0) + 1
        option_counts[scenario.option] = option_counts.get(scenario.option, 0) + 1
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
        "option_episode_counts": option_counts,
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
        "action_abs_mean": mean(action_values),
        "action_saturation_rate": mean([float(value) for value in saturated]),
        "training_signal": training_signal,
    }
    return metrics


def _training_signal_metrics(
    records: list[tuple[ScenarioMiningTaskSpec, Any]],
) -> dict[str, Any]:
    """Expose the event and reward density available to Inner SAC."""
    buckets: dict[str, dict[str, int]] = {}

    def bucket(name: str) -> dict[str, int]:
        return buckets.setdefault(name, {
            "episodes": 0,
            "valid_event_episodes": 0,
            "valid_target_collision_episodes": 0,
            "valid_near_miss_episodes": 0,
            "transitions": 0,
            "positive_reward_transitions": 0,
            "event_capture_transitions": 0,
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
            values["positive_reward_transitions"] += int(float(row["reward_inner"]) > 0.0)
            values["event_capture_transitions"] += int(
                bool(row["info"].get("event_just_captured", False))
            )

    for task, episode in records:
        add("overall", episode)
        add(f"family:{task.functional_scenario}", episode)
        add(f"option:{episode.concrete_scenario.option}", episode)

    return {
        name: {
            **values,
            "positive_reward_transition_fraction": float(
                values["positive_reward_transitions"] / max(values["transitions"], 1)
            ),
        }
        for name, values in sorted(buckets.items())
    }


def train_posterior(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    settings = dict(config["posterior"])
    support_choices = tuple(sorted({int(value) for value in settings["support_episodes"]}))
    budget = int(settings["episode_budget"])
    remaining = budget
    if not support_choices or min(support_choices) < 1 or budget < 2:
        raise ValueError("posterior requires positive support K and a held-out target")
    losses, counts, sampler = [], [], MetaTaskSampler(tasks)
    while remaining:
        choices = [
            count
            for count in support_choices
            if count + 1 <= remaining and remaining - count - 1 != 1
        ]
        if not choices:
            raise ValueError("posterior episode_budget cannot be partitioned into held-out K groups")
        support_count = random.choice(choices)
        task = sampler.sample()
        result = build_online(model, task, int(config["training"]["step_budget"]), criteria).run(
            task, support_count + 1, posterior_support_limit=0
        )
        loss = model.posterior_loss(posterior_batch_from_episodes(model, result.episodes))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        counts.append(support_count)
        remaining -= support_count + 1
    return {
        "simulator_episodes_consumed": budget,
        "optimizer_updates": len(losses),
        "support_counts": counts,
        "posterior_loss": float(np.mean(losses)),
    }


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
