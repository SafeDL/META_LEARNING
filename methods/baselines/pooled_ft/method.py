"""Pooled pointwise supervision, deliberately distinct from task-level GP evidence."""

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.common.response import archive_fingerprints
from methods.common.model_contract import model_contract, validate_model_contract


CONFIG = {"steps": 1500, "batch_size": 128, "learning_rate": 0.001,
          "validation_interval": 100, "support_count": 8, "query_count": 32,
          "adaptation_steps": 20, "adaptation_rate": 0.001, "seed": 1409}


class CollisionNetwork(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.network = nn.Sequential(nn.Linear(dimension, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1)


def fine_tune(model, x, y, *, steps=20, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    labels = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), labels)
        loss.backward()
        optimizer.step()


def train(history, development, destination, *, held_family=None, config=CONFIG):
    if not history or not development or any(a.role != "history" for a in history):
        raise ValueError("Pooled pretraining uses historical objects only")
    if any(a.role != "development" for a in development):
        raise ValueError("Development objects must select the pooled checkpoint")
    if held_family and any(a.family == held_family for a in (*history, *development)):
        raise ValueError("Held family leaked into pooled pretraining")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contract = model_contract("pooled_ft")
    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    generator = np.random.default_rng(config["seed"])
    model = CollisionNetwork(history[0].x.shape[1])
    x = np.concatenate([a.x for a in history])
    y = np.concatenate([a.collision for a in history])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, curve = float("inf"), []
    for step in range(1, config["steps"] + 1):
        indices = generator.choice(len(x), min(config["batch_size"], len(x)), replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x[indices]), torch.as_tensor(y[indices], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        if step % config["validation_interval"] == 0 or step == config["steps"]:
            scores = []
            for target in development:
                indices = np.random.default_rng(config["seed"] + 1).choice(
                    len(target.scenarios), config["support_count"] + config["query_count"], replace=False)
                support, query = indices[:config["support_count"]], indices[config["support_count"]:]
                adapted = copy.deepcopy(model)
                fine_tune(adapted, target.x[support], target.collision[support],
                          steps=config["adaptation_steps"], learning_rate=config["adaptation_rate"])
                with torch.no_grad():
                    scores.append(float(F.binary_cross_entropy_with_logits(
                        adapted(target.x[query]), torch.as_tensor(target.collision[query], dtype=torch.float32))))
            score = float(np.mean(scores))
            curve.append({"step": step, "development_nll": score})
            if score < best:
                best = score
                torch.save({"format": "pooled-collision", "dimension": model.dimension,
                            "model_contract": contract,
                            "state_dict": model.state_dict(), "config": config,
                            "history_ids": [a.source_id for a in history],
                            "development_ids": [a.source_id for a in development],
                            "history_fingerprints": archive_fingerprints(history),
                            "development_fingerprints": archive_fingerprints(development),
                            "held_family": held_family}, destination)
    report = {"status": "complete", "curve": curve, "config": config,
              "model_contract": contract,
              "elapsed_s": time.perf_counter() - started, "held_family": held_family}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["format"] != "pooled-collision":
        raise ValueError("Pooled baseline needs a new collision checkpoint")
    validate_model_contract(checkpoint, "pooled_ft")
    model = CollisionNetwork(checkpoint["dimension"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
