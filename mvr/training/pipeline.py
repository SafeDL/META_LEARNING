"""The only supported MVR training sequence and checkpoint contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from ..failure.criteria import FailureCriteria
from ..model import TransferableScenarioMiner
from ..provenance import content_hash
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.task_spec import ScenarioMiningTaskSpec
from ..scenario.taskbook import load_taskbook
from .checkpoint import HierarchicalCheckpoint
from .stages import CANONICAL_TRAINING_STAGES, TrainingStage, validate_stage_transition
from .trainers import train_inner, train_inner_latent_calibration, train_outer, train_posterior
from .workflow import StagedWorkflow


CHECKPOINT_CONFIG_KEYS = (
    "schema",
    "taskbook",
    "model",
    "context",
    "map",
    "failure",
    "inner",
    "posterior",
    "inner_latent_calibration",
    "outer",
    "training",
)
CANONICAL_STAGES = CANONICAL_TRAINING_STAGES


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


def checkpoint_config_hash(config: dict[str, Any]) -> str:
    return content_hash({key: config[key] for key in CHECKPOINT_CONFIG_KEYS if key in config})


def taskbook_hash(taskbook: str | Path) -> str:
    return content_hash(json.loads(Path(taskbook).read_text(encoding="utf-8")))


def assert_taskbook_compatible(checkpoint: HierarchicalCheckpoint, taskbook: str | Path) -> None:
    if checkpoint.state.get("taskbook_hash") != taskbook_hash(taskbook):
        raise ValueError("checkpoint taskbook hash mismatch")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path, torch.device]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    taskbook = Path(config["taskbook"])
    if not taskbook.is_file():
        raise FileNotFoundError(f"taskbook not found: {taskbook}")
    device = resolve_device(str(config["training"].get("device", "cuda")))
    seed_everything(int(config["seed"]))
    return config, taskbook, device


def selected_tasks(
    config: dict[str, Any],
    taskbook: Path,
    sut_split: str,
    geometry_split: str,
    functional_split: str = "train",
) -> list[ScenarioMiningTaskSpec]:
    family = str(config["training"].get("family_filter", "all"))
    tasks = [
        task
        for task in load_taskbook(taskbook)
        if task.sut_split == sut_split
        and task.geometry_split == geometry_split
        and task.functional_split == functional_split
        and (family == "all" or task.functional_scenario == family)
    ]
    if not tasks:
        raise ValueError("taskbook has no tasks for the requested OOD axes")
    return tasks


def build_model(config: dict[str, Any], device: torch.device) -> TransferableScenarioMiner:
    space = next(iter(mvr_parameter_spaces().values()))
    return TransferableScenarioMiner(
        state_dim=int(config["model"].get("state_dim", 10)),
        map_dim=int(config["map"].get("embedding_dim", 128)),
        latent_dim=int(config["model"].get("latent_dim", 16)),
        continuous_dim=space.continuous_dim,
        option_count=len(space.options),
        num_experts=int(config["model"].get("num_experts", 4)),
        context_kl_weight=float(config.get("context", {}).get("kl_weight", 1e-3)),
    ).to(device)


def _optimizer(
    model: TransferableScenarioMiner,
    components: Iterable[str],
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = [
        parameter
        for name in sorted(components)
        for parameter in model.training_components()[name].parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("active stage has no trainable parameters")
    return torch.optim.Adam(parameters, lr=learning_rate)


def _stage_settings(config: dict[str, Any], stage: TrainingStage) -> dict[str, Any]:
    sections = {
        TrainingStage.INNER_PRETRAIN: "inner",
        TrainingStage.POSTERIOR: "posterior",
        TrainingStage.INNER_LATENT_CALIBRATION: "inner_latent_calibration",
        TrainingStage.OUTER: "outer",
    }
    return config[sections[stage]]


def _save_stage(
    path: Path,
    stage: TrainingStage,
    config: dict[str, Any],
    model: TransferableScenarioMiner,
    metrics: dict[str, Any],
    device: torch.device,
    taskbook: Path,
) -> None:
    taskbook_digest = taskbook_hash(taskbook)
    state = {
        "model": model.state_dict(),
        "provenance_config": config,
        "taskbook_hash": taskbook_digest,
    }
    compatibility_hash = checkpoint_config_hash(config)
    HierarchicalCheckpoint(HierarchicalCheckpoint.SCHEMA, stage.value, compatibility_hash, state).save(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "stage": stage.value,
                "device": str(device),
                "seed": config["seed"],
                "compatibility_hash": compatibility_hash,
                "taskbook_hash": taskbook_digest,
                "config": config,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


class MVRTrainingPipeline:
    def __init__(self, config: dict[str, Any], taskbook: Path, device: torch.device) -> None:
        self.config = config
        self.taskbook = taskbook
        self.device = device
        self.criteria = FailureCriteria.from_config(config["failure"])
        self.model = build_model(config, device)
        self.workflow = StagedWorkflow(self.model.training_components())

    def _load_resume(self, path: str | Path | None) -> int:
        if path is None:
            return 0
        checkpoint = HierarchicalCheckpoint.load(
            path, expected_config_hash=checkpoint_config_hash(self.config)
        )
        try:
            stage = TrainingStage(checkpoint.stage)
        except ValueError as error:
            raise ValueError("resume checkpoint is not a canonical MVR stage") from error
        if stage not in CANONICAL_STAGES:
            raise ValueError("resume checkpoint is not a canonical MVR stage")
        self.model.load_state_dict(checkpoint.state["model"])
        assert_taskbook_compatible(checkpoint, self.taskbook)
        return CANONICAL_STAGES.index(stage) + 1

    def _train_stage(
        self,
        stage: TrainingStage,
        tasks: list[ScenarioMiningTaskSpec],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        trainers = {
            TrainingStage.INNER_PRETRAIN: train_inner,
            TrainingStage.POSTERIOR: train_posterior,
            TrainingStage.INNER_LATENT_CALIBRATION: train_inner_latent_calibration,
            TrainingStage.OUTER: train_outer,
        }
        result = trainers[stage](self.model, tasks, self.config, self.criteria, optimizer)
        return result[0] if isinstance(result, tuple) else result

    def run(self, output: str | Path, *, resume: str | Path | None = None) -> Path:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        tasks = selected_tasks(self.config, self.taskbook, "train", "train")
        start = self._load_resume(resume)
        if start == len(CANONICAL_STAGES):
            raise ValueError("the resume checkpoint already completed the canonical pipeline")
        records = []
        predecessor = Path(resume) if resume is not None else None
        for index, stage in enumerate(CANONICAL_STAGES[start:], start=start):
            previous_stage = None
            if predecessor is not None:
                previous_stage = TrainingStage(HierarchicalCheckpoint.load(predecessor).stage)
            validate_stage_transition(stage, previous_stage)
            stage_seed = int(self.config["seed"]) + index
            seed_everything(stage_seed)
            active = self.workflow.activate(stage)
            settings = _stage_settings(self.config, stage)
            optimizer = _optimizer(
                self.model,
                active,
                float(settings.get("learning_rate", self.config["training"]["learning_rate"])),
            )
            metrics = self._train_stage(stage, tasks, optimizer)
            checkpoint_path = output_path / f"{stage.value}.pt"
            _save_stage(
                checkpoint_path,
                stage,
                self.config,
                self.model,
                metrics,
                self.device,
                self.taskbook,
            )
            predecessor = checkpoint_path
            records.append(
                {
                    "stage": stage.value,
                    "seed": stage_seed,
                    "checkpoint": checkpoint_path.name,
                    "metrics": metrics,
                }
            )
        manifest = output_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "stages": records,
                    "compatibility_hash": checkpoint_config_hash(self.config),
                    "taskbook": str(self.taskbook),
                    "taskbook_hash": taskbook_hash(self.taskbook),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest


def train(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the canonical MVR training pipeline.")
    parser.add_argument("--config", default="mvr/configs/mvr.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config, taskbook, device = load_config(args.config)
    MVRTrainingPipeline(config, taskbook, device).run(args.output, resume=args.resume)
