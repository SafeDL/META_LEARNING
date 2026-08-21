"""Shared minimal stage entry point: reproducible ownership and checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from ..model import HierarchicalMetaTester
from ..provenance import content_hash
from ..scenario.catalog import mvr_parameter_spaces
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.stages import TrainingStage
from ..training.workflow import StagedWorkflow


def run(stage: TrainingStage, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="meta_testing/configs/mvr.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed, device = int(config["seed"]), torch.device(config.get("training", {}).get("device", "cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = HierarchicalMetaTester(mvr_parameter_spaces(), state_dim=int(config.get("model", {}).get("state_dim", 5))).to(device)
    workflow = StagedWorkflow(model.training_components())
    active = workflow.activate(stage)
    config_hash = content_hash(config)
    if args.resume:
        checkpoint = HierarchicalCheckpoint.load(args.resume, expected_config_hash=config_hash)
        model.load_state_dict(checkpoint.state["model"])
    checkpoint = HierarchicalCheckpoint(HierarchicalCheckpoint.SCHEMA, stage.value, config_hash, {"model": model.state_dict()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.save(output)
    output.with_suffix(".json").write_text(json.dumps({"stage": stage.value, "seed": seed, "device": str(device), "trainable_components": sorted(active), "family_filter": config.get("training", {}).get("family_filter", "all"), "profile_split": config.get("sut_profiles", {}), "episode_budget": config.get("training", {}).get("episode_budget"), "step_budget": config.get("training", {}).get("step_budget")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
