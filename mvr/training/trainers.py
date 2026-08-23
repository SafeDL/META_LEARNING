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
from .updates import update_inner_sac, update_outer_ppo


def build_online(
    model: TransferableScenarioMiner,
    task: ScenarioMiningTaskSpec,
    max_steps: int,
    criteria: FailureCriteria,
) -> OnlineMetaTest:
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    return OnlineMetaTest(model, executor, HierarchicalRunner(max_steps, criteria))


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
            losses.append(update_inner_sac(model, replay, optimizer, batch_size=batch_size))


def train_inner(
    model: TransferableScenarioMiner,
    tasks: list[ScenarioMiningTaskSpec],
    config: dict[str, Any],
    criteria: FailureCriteria,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], InnerReplay]:
    settings = dict(config["inner"])
    budget, max_steps = int(settings["episode_budget"]), int(config["training"]["step_budget"])
    replay, losses = _replay(settings, None), []
    sampler = MetaTaskSampler(tasks)
    for _ in range(budget):
        task = sampler.sample()
        result = build_online(model, task, max_steps, criteria).run(task, 1)
        for row in result.inner_transitions:
            replay.add(row)
        _update_inner(model, replay, optimizer, settings, losses)
    return _inner_metrics(budget, replay, losses), replay


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
    episodes: int, replay: InnerReplay, losses: list[dict[str, float]]
) -> dict[str, Any]:
    return {
        "simulator_episodes_consumed": episodes,
        "transitions": len(replay.rows),
        "optimizer_updates": len(losses),
        "last_loss": losses[-1] if losses else {},
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
