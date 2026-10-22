"""SUT-held-out Bernoulli episodes; independent development checkpoint selection."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from methods.common.classification import torch_condition, torch_moments
from methods.common.response import ResponseEncoder, archive_fingerprints
from methods.common.model_contract import model_contract, validate_model_contract

from .config import CONFIG
from .prior import component_prior


def episode(archives, target, generator, support_count, query_count):
    contexts = tuple(a for a in archives if a.source_id != target.source_id)
    if not contexts or any(a.source_id == target.source_id for a in contexts):
        raise ValueError("The simulated target cannot appear in its own response context")
    if any(a.role != "history" for a in contexts):
        raise ValueError("Only historical sources form episodic contexts")
    indices = generator.choice(len(target.scenarios), support_count + query_count, replace=False)
    return contexts, indices


def episode_loss(encoder, contexts, target, indices, support_count, config=CONFIG):
    x = target.x[indices]
    labels = target.collision[indices]
    mean, covariance, weight = component_prior(
        encoder, x, contexts, target_weight=config.target_weight)
    log_weight = weight.log()
    for index in range(support_count):
        mean, covariance, evidence = torch_condition(
            mean, covariance, index, bool(labels[index]))
        log_weight = log_weight + evidence
        log_weight = log_weight - torch.logsumexp(log_weight, 0)
    probability = (log_weight.exp()[:, None] * torch_moments(
        mean[:, support_count:], covariance.diagonal(dim1=1, dim2=2)[:, support_count:])[0]).sum(0)
    truth = torch.as_tensor(labels[support_count:], dtype=probability.dtype,
                            device=probability.device)
    classification = F.binary_cross_entropy(probability.clamp(1e-8, 1 - 1e-8), truth)
    auxiliary_truth = encoder.tensor(target.auxiliary[indices[support_count:]])
    auxiliary = F.mse_loss(encoder.auxiliary_prediction(x[support_count:], contexts),
                           auxiliary_truth)
    return classification + config.auxiliary_weight * auxiliary, {
        "collision_nll": float(classification.detach()), "auxiliary_mse": float(auxiliary.detach()),
        "brier": float(((probability - truth)**2).mean().detach())}


def validate(encoder, history, development, config=CONFIG):
    encoder.eval()
    rows = []
    with torch.no_grad():
        for target in development:
            generator = np.random.default_rng(config.seed + 1)
            contexts, indices = episode(history, target, generator,
                                         max(config.support_counts), config.query_count)
            _, metrics = episode_loss(encoder, contexts, target, indices,
                                       max(config.support_counts), config)
            rows.append({"family": target.family, **metrics})
    encoder.train()
    # Families are the outer units; repetitions within one family do not add weight.
    families = sorted({r["family"] for r in rows})
    return {key: float(np.mean([np.mean([r[key] for r in rows if r["family"] == f])
                               for f in families]))
            for key in ("collision_nll", "auxiliary_mse", "brier")}


def train(history, development, destination, *, held_family=None, config=CONFIG):
    if not history or not development or any(a.role != "history" for a in history):
        raise ValueError("Training requires explicit historical and development archives")
    if any(a.role != "development" for a in development):
        raise ValueError("Only independent development tasks select checkpoints")
    if held_family and any(a.family == held_family for a in (*history, *development)):
        raise ValueError("Held family leaked into training or development")
    if len({a.source_id for a in (*history, *development)}) != len(history) + len(development):
        raise ValueError("Archive source identities overlap")
    kinds = {a.scenarios[0].kind for a in (*history, *development)}
    if len(kinds) != 1:
        raise ValueError("Train one functional scenario model at a time")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Model already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    contract = model_contract("history_response")
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ResponseEncoder(history[0].x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.learning_rate, weight_decay=0)
    best = (float("inf"), float("inf"))
    curve = []
    for step in range(1, config.steps + 1):
        target = history[(step - 1) % len(history)]
        support = int(generator.choice(config.support_counts))
        contexts, indices = episode(history, target, generator, support, config.query_count)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = episode_loss(encoder, contexts, target, indices, support, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite collision training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        optimizer.step()
        if step % config.validation_interval == 0 or step == config.steps:
            validation = validate(encoder, history, development, config)
            score = (validation["collision_nll"], validation["brier"])
            curve.append({"step": step, "training": metrics, "validation": validation})
            if score < best:
                best = score
                torch.save({"format": "collision-bernoulli", "state_dict": encoder.state_dict(),
                            "model_contract": contract,
                            "dimension": encoder.dimension, "step": step, "config": asdict(config),
                            "kind": next(iter(kinds)), "held_family": held_family,
                            "history_ids": [a.source_id for a in history],
                            "development_ids": [a.source_id for a in development],
                            "history_fingerprints": archive_fingerprints(history),
                            "development_fingerprints": archive_fingerprints(development)}, destination)
            print(f"collision model step={step} validation_nll={score[0]:.6f}", flush=True)
    report = {"status": "complete", "config": asdict(config), "curve": curve,
              "model_contract": contract,
              "elapsed_s": time.perf_counter() - started, "held_family": held_family,
              "history_ids": [a.source_id for a in history],
              "development_ids": [a.source_id for a in development]}
    destination.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def load_encoder(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "collision-bernoulli":
        raise ValueError("This is not a collision-supervised model")
    validate_model_contract(checkpoint, "history_response")
    encoder = ResponseEncoder(checkpoint["dimension"])
    encoder.load_state_dict(checkpoint["state_dict"])
    return encoder.eval().requires_grad_(False), checkpoint
