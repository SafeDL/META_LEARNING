"""Task-batch GP evidence learning and target deep-kernel fine-tuning."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from testing_methods.common.gp_classifier import (
    ClassificationKernel,
    collision_probabilities,
    fit_target,
    laplace,
)
from testing_methods.common.model_contract import model_contract, validate_model_contract
from testing_methods.common.response import archive_fingerprints

CONFIG = {
    "steps": 1500,
    "batch_size": 64,
    "learning_rate": 0.001,
    "validation_interval": 100,
    "support_count": 8,
    "query_count": 32,
    "adaptation_steps": 10,
    "adaptation_rate": 0.003,
    "seed": 1301,
}


def train(history, development, destination, *, held_family=None, config=CONFIG):
    if not history or not development or any(a.role != "history" for a in history):
        raise ValueError("FSBO trains only on historical task archives")
    if any(a.role != "development" for a in development):
        raise ValueError("FSBO checkpoint selection requires separate development objects")
    if held_family and any(a.family == held_family for a in (*history, *development)):
        raise ValueError("Held family leaked into FSBO construction")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contract = model_contract("fsbo")
    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    generator = np.random.default_rng(config["seed"])
    model = ClassificationKernel(history[0].x.shape[1], deep=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, curve = float("inf"), []
    for step in range(1, config["steps"] + 1):
        # A batch belongs to one SUT: task-specific latent GP values are integrated out.
        task = history[int(generator.integers(len(history)))]
        indices = generator.choice(
            len(task.scenarios), min(config["batch_size"], len(task.scenarios)), replace=False
        )
        optimizer.zero_grad(set_to_none=True)
        loss = laplace(model, task.x[indices], task.collision[indices])[0] / len(indices)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite FSBO task evidence")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % config["validation_interval"] == 0 or step == config["steps"]:
            scores = []
            for target in development:
                indices = np.random.default_rng(config["seed"] + 1).choice(
                    len(target.scenarios),
                    config["support_count"] + config["query_count"],
                    replace=False,
                )
                support, query = (
                    indices[: config["support_count"]],
                    indices[config["support_count"] :],
                )
                adapted = copy.deepcopy(model)
                fit_target(
                    adapted,
                    target.x[support],
                    target.collision[support],
                    steps=config["adaptation_steps"],
                    learning_rate=config["adaptation_rate"],
                )
                with torch.no_grad():
                    p = collision_probabilities(
                        adapted, target.x[support], target.collision[support], target.x[query]
                    )
                    y = torch.as_tensor(target.collision[query], dtype=p.dtype)
                    scores.append(float(torch.nn.functional.binary_cross_entropy(p, y)))
            score = float(np.mean(scores))
            curve.append({"step": step, "development_nll": score})
            if score < best:
                best = score
                torch.save(
                    {
                        "format": "fsbo-collision",
                        "dimension": model.dimension,
                        "model_contract": contract,
                        "state_dict": model.state_dict(),
                        "config": config,
                        "history_ids": [a.source_id for a in history],
                        "development_ids": [a.source_id for a in development],
                        "history_fingerprints": archive_fingerprints(history),
                        "development_fingerprints": archive_fingerprints(development),
                        "held_family": held_family,
                    },
                    destination,
                )
    report = {
        "status": "complete",
        "curve": curve,
        "elapsed_s": time.perf_counter() - started,
        "model_contract": contract,
        "config": config,
        "held_family": held_family,
        "adaptation": "Bernoulli task evidence and collision-probability acquisition; no label rescaling",
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["format"] != "fsbo-collision":
        raise ValueError("FSBO needs a newly meta-trained collision checkpoint")
    validate_model_contract(checkpoint, "fsbo")
    model = ClassificationKernel(checkpoint["dimension"], deep=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
