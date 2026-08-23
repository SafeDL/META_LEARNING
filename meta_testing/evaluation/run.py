"""Fixed-budget evaluation for a trained canonical MVR checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..failure.criteria import FailureCriteria
from ..failure.metrics import FixedBudgetMetrics
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.online_meta_test import OnlineMetaTestResult
from ..training.pipeline import (
    assert_taskbook_compatible,
    build_model,
    checkpoint_config_hash,
    load_config,
    seed_everything,
    selected_tasks,
)
from ..training.stages import TrainingStage
from ..training.trainers import build_online
from .budget_protocol import BudgetProtocol


def evaluate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVR under the configured fixed budget.")
    parser.add_argument("--config", default="meta_testing/configs/mvr.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config, taskbook, device = load_config(args.config)
    checkpoint = HierarchicalCheckpoint.load(args.checkpoint, expected_config_hash=checkpoint_config_hash(config))
    if checkpoint.stage != TrainingStage.OUTER.value:
        raise ValueError("formal evaluation requires an outer checkpoint")
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
    criteria = FailureCriteria.from_config(config["failure"])
    results: dict[str, Any] = {}
    for task in selected_tasks(config, taskbook, "meta_test", "meta_test"):
        by_shot: dict[str, list[dict[str, Any]]] = {str(shot): [] for shot in protocol.support_shots}
        for seed in settings["seeds"]:
            seed_everything(int(seed))
            for shot in protocol.support_shots:
                online_result: OnlineMetaTestResult = build_online(
                    model, task, int(config["training"]["step_budget"]), criteria
                ).run(
                    task,
                    protocol.total_episode_budget,
                    deterministic=bool(settings["deterministic"]),
                    posterior_support_limit=shot,
                )
                metrics = FixedBudgetMetrics(protocol.total_episode_budget)
                for episode in online_result.episodes:
                    metrics.add(episode.rollout.signature)
                by_shot[str(shot)].append({"seed": int(seed), **metrics.summary()})
        results[task.task_id] = {
            shot: {
                "runs": runs,
                "mean_failure_discovery_auc": float(np.mean([run["failure_discovery_auc"] for run in runs])),
                "mean_unique_failures": float(np.mean([run["cumulative_unique_failures"] for run in runs])),
            }
            for shot, runs in by_shot.items()
        }
    Path(args.output).write_text(
        json.dumps({"checkpoint_stage": checkpoint.stage, "device": str(device), "tasks": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
