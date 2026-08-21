"""Shared real training and evaluation entry points for MVR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from ..evaluation.budget_protocol import BudgetProtocol
from ..failure.analyzer import analyze_rollout
from ..failure.metrics import FixedBudgetMetrics
from ..model import HierarchicalMetaTester
from ..provenance import content_hash
from ..scenario.adapters import CutInScenarioAdapter, MergeScenarioAdapter, RoundaboutScenarioAdapter
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.task_spec import MetaTestTaskSpec
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.online_meta_test import OnlineMetaTest, OnlineMetaTestResult
from ..training.posterior_data import posterior_batch_from_episodes
from ..training.replay import InnerReplay
from ..training.runner import HierarchicalRunner
from ..training.stages import TrainingStage
from ..training.updates import update_inner_sac, update_outer_ppo
from ..training.workflow import StagedWorkflow


ADAPTERS = {"merge": MergeScenarioAdapter(), "cutin": CutInScenarioAdapter(), "roundabout": RoundaboutScenarioAdapter()}


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("training.device must be 'cuda' or 'cpu'")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in state["cuda"]])


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _parser(*, evaluation: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="meta_testing/configs/mvr.yaml")
    parser.add_argument("--taskbook")
    parser.add_argument("--output", required=True)
    if evaluation:
        parser.add_argument("--checkpoint", required=True)
    else:
        parser.add_argument("--resume")
    return parser


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path, torch.device]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    taskbook_value = args.taskbook or config.get("taskbook")
    if not taskbook_value:
        raise ValueError("provide a taskbook in the YAML config or with --taskbook")
    taskbook = Path(taskbook_value)
    if not taskbook.is_file():
        raise FileNotFoundError(f"taskbook not found: {taskbook}")
    device = resolve_device(str(config.get("training", {}).get("device", "cuda")))
    seed_everything(int(config["seed"]))
    return config, taskbook, device


def _tasks(config: dict[str, Any], taskbook: Path, split: str, profiles_key: str) -> list[MetaTestTaskSpec]:
    training = config.get("training", {})
    family = str(training.get("family_filter", "all"))
    profiles = set(config.get("sut_profiles", {}).get(profiles_key, ()))
    tasks = [task for task in load_taskbook(taskbook) if task.split == split and (family == "all" or task.scenario_family == family) and (not profiles or task.sut_ref in profiles)]
    if not tasks:
        raise ValueError(f"taskbook has no selected {split} tasks for the configured family/profile split")
    return tasks


def _model(config: dict[str, Any], device: torch.device) -> HierarchicalMetaTester:
    model = HierarchicalMetaTester(
        mvr_parameter_spaces(),
        state_dim=int(config.get("model", {}).get("state_dim", 5)),
        map_dim=int(config.get("map", {}).get("embedding_dim", 128)),
    )
    return model.to(device)


def _optimizer(model: HierarchicalMetaTester, components: Iterable[str], learning_rate: float) -> torch.optim.Optimizer:
    parameters = [parameter for name in components for parameter in model.training_components()[name].parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("active stage has no trainable parameters")
    return torch.optim.Adam(parameters, lr=learning_rate)


def _online(model: HierarchicalMetaTester, task: MetaTestTaskSpec, max_steps: int) -> OnlineMetaTest:
    executor = ScenarioExecutor(ADAPTERS, mvr_parameter_spaces())
    return OnlineMetaTest(model, executor, HierarchicalRunner(max_steps=max_steps), lambda transitions: analyze_rollout(transitions, task.scenario_family))


def _restore(model: HierarchicalMetaTester, optimizer: torch.optim.Optimizer, path: str | None, config_hash: str, device: torch.device, stage: TrainingStage) -> dict[str, Any]:
    if not path:
        return {}
    checkpoint = HierarchicalCheckpoint.load(path, expected_config_hash=config_hash)
    model.load_state_dict(checkpoint.state["model"])
    if checkpoint.stage == stage.value and "optimizer" in checkpoint.state:
        optimizer.load_state_dict(checkpoint.state["optimizer"])
        _move_optimizer_state(optimizer, device)
    if "rng_state" in checkpoint.state:
        _restore_rng(dict(checkpoint.state["rng_state"]))
    return dict(checkpoint.state.get("progress", {}))


def _save(path: Path, *, stage: str, config: dict[str, Any], model: HierarchicalMetaTester, optimizer: torch.optim.Optimizer | None, progress: dict[str, Any], metrics: dict[str, Any], device: torch.device) -> None:
    state: dict[str, Any] = {"model": model.state_dict(), "progress": progress, "rng_state": _rng_state()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    HierarchicalCheckpoint(HierarchicalCheckpoint.SCHEMA, stage, content_hash(config), state).save(path)
    path.with_suffix(".json").write_text(json.dumps({"stage": stage, "device": str(device), "seed": config["seed"], "metrics": metrics, "progress": progress}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _train_inner(model: HierarchicalMetaTester, tasks: list[MetaTestTaskSpec], config: dict[str, Any], optimizer: torch.optim.Optimizer, start_episode: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config["training"]
    budget, max_steps = int(settings["episode_budget"]), int(settings["step_budget"])
    batch_size = int(config.get("inner", {}).get("batch_size", 64))
    replay, losses = InnerReplay(), []
    for index in range(start_episode, start_episode + budget):
        task = tasks[index % len(tasks)]
        result = _online(model, task, max_steps).run(task, 1)
        for row in result.inner_transitions:
            replay.add(row)
        if len(replay.rows) >= batch_size:
            losses.append(update_inner_sac(model, replay, optimizer, batch_size=batch_size))
    return {"episodes": budget, "transitions": len(replay.rows), "updates": len(losses), "last_loss": losses[-1] if losses else {}}, {"episodes": start_episode + budget}


def _train_posterior(model: HierarchicalMetaTester, tasks: list[MetaTestTaskSpec], config: dict[str, Any], optimizer: torch.optim.Optimizer, start_episode: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config["training"]
    support_count = int(config.get("posterior", {}).get("support_episodes", 1))
    group_size, budget = support_count + 1, int(settings["episode_budget"])
    if budget < group_size:
        raise ValueError("posterior episode_budget must cover support plus one held-out target")
    losses = []
    for index in range(start_episode // group_size, start_episode // group_size + budget // group_size):
        task = tasks[index % len(tasks)]
        result = _online(model, task, int(settings["step_budget"])).run(task, group_size)
        loss = model.posterior_loss(posterior_batch_from_episodes(model, result.episodes))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    episodes = len(losses) * group_size
    return {"episodes": episodes, "updates": len(losses), "posterior_loss": float(np.mean(losses)) if losses else None}, {"episodes": start_episode + episodes}


def _train_outer(model: HierarchicalMetaTester, tasks: list[MetaTestTaskSpec], config: dict[str, Any], optimizer: torch.optim.Optimizer, start_episode: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config["training"]
    budget, max_steps = int(settings["episode_budget"]), int(settings["step_budget"])
    losses, episodes = [], 0
    for task in tasks:
        result = _online(model, task, max_steps).run(task, budget)
        losses.append(update_outer_ppo(model.scene_policies[task.parameter_space_id], result.outer_rollout, optimizer, epochs=int(config.get("outer", {}).get("ppo_epochs", 4)), batch_size=int(config.get("outer", {}).get("batch_size", 64))))
        episodes += len(result.episodes)
    return {"episodes": episodes, "updates": len(losses), "outer_ppo_loss": float(np.mean(losses)) if losses else None}, {"episodes": start_episode + episodes}


def run(stage: TrainingStage, argv: list[str] | None = None) -> None:
    args = _parser(evaluation=False).parse_args(argv)
    config, taskbook, device = _load_config(args)
    model = _model(config, device)
    workflow = StagedWorkflow(model.training_components())
    active = workflow.activate(stage)
    section = {TrainingStage.INNER_PRETRAIN: "inner", TrainingStage.POSTERIOR: "posterior", TrainingStage.OUTER: "outer", TrainingStage.LIGHT_JOINT: "light_joint"}[stage]
    optimizer = _optimizer(model, active, float(config.get(section, {}).get("learning_rate", config.get("training", {}).get("learning_rate", 3e-4))))
    previous = _restore(model, optimizer, args.resume, content_hash(config), device, stage)
    workflow.activate(stage)
    tasks = _tasks(config, taskbook, "meta_train", "meta_train")
    trainer = {TrainingStage.INNER_PRETRAIN: _train_inner, TrainingStage.POSTERIOR: _train_posterior, TrainingStage.OUTER: _train_outer}.get(stage)
    if trainer is None:
        raise ValueError("light joint calibration is deliberately not a default training entry point")
    metrics, progress = trainer(model, tasks, config, optimizer, int(previous.get("episodes", 0)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save(output, stage=stage.value, config=config, model=model, optimizer=optimizer, progress=progress, metrics=metrics, device=device)


def evaluate(argv: list[str] | None = None) -> None:
    args = _parser(evaluation=True).parse_args(argv)
    config, taskbook, device = _load_config(args)
    checkpoint = HierarchicalCheckpoint.load(args.checkpoint, expected_config_hash=content_hash(config))
    model = _model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    budget = int(config.get("evaluation", {}).get("total_episode_budget", config["training"]["episode_budget"]))
    results: dict[str, Any] = {}
    for task in _tasks(config, taskbook, "meta_test", "meta_test"):
        BudgetProtocol(budget, tuple(config.get("evaluation", {}).get("support_shots", (0,)))).validate()
        online_result: OnlineMetaTestResult = _online(model, task, int(config["training"]["step_budget"])).run(task, budget, deterministic=True)
        metrics = FixedBudgetMetrics(budget)
        for episode in online_result.episodes:
            metrics.add(episode.rollout.signature)
        results[task.task_id] = metrics.summary()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint_stage": checkpoint.stage, "device": str(device), "tasks": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
