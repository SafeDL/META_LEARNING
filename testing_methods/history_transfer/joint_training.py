"""Historical task-evidence learning for paired collision/clearance responses."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.discovery_config import read_json, write_json
from experiments.protocol import ROOT
from experiments.source_registration import source_contract_matches, specification_matches
from testing_methods.common.response import archive_fingerprints

from .joint_response import (
    JointResponseKernel,
    fit_joint,
    joint_loss,
    margin_values,
    posterior,
    probabilities,
)

TRAINING = {
    "steps": 1500,
    "learning_rate": 0.001,
    "seed": 1301,
    "support_counts": (8, 16, 32, 64, 128),
    "validation_interval": 100,
    "validation_seed": 1319,
    "query_count": 32,
    "adaptation_steps": 10,
    "adaptation_rate": 0.003,
}


def training_contract():
    files = (
        "testing_methods/history_transfer/joint_training.py",
        "testing_methods/history_transfer/joint_response.py",
        "testing_methods/common/gp_classifier.py",
        "testing_methods/common/classification.py",
        "testing_methods/common/response.py",
        "experiments/schema.py",
        "experiments/scenarios.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def validate(model, development, config):
    model.eval()
    rows = []
    maximum, query_count = max(config["support_counts"]), config["query_count"]
    generator = np.random.default_rng(config["validation_seed"])
    for archive in development:
        indices = generator.choice(len(archive.scenarios), maximum + query_count, replace=False)
        x, labels, clearance = (
            archive.x[indices],
            archive.collision[indices],
            archive.safety[indices, 0],
        )
        for count in config["support_counts"]:
            adapted = copy.deepcopy(model)
            fit_joint(
                adapted,
                x[:count],
                labels[:count],
                clearance[:count],
                steps=config["adaptation_steps"],
                learning_rate=config["adaptation_rate"],
            )
            with torch.no_grad():
                _, mean, covariance = posterior(
                    adapted.joint_mean(len(x)),
                    adapted.joint_covariance(x),
                    range(count),
                    labels[:count],
                    margin_values(clearance[:count]),
                )
                probability = probabilities(mean, covariance)[-query_count:].clamp(1e-8, 1 - 1e-8)
                truth = torch.as_tensor(
                    labels[-query_count:], dtype=probability.dtype, device=probability.device
                )
                margin_truth = torch.as_tensor(
                    margin_values(clearance[-query_count:]), dtype=mean.dtype, device=mean.device
                )
                rows.append(
                    {
                        "source_id": archive.source_id,
                        "lineage": archive.lineage,
                        "support_count": count,
                        "collision_nll": float(F.binary_cross_entropy(probability, truth)),
                        "brier": float((probability - truth).square().mean()),
                        "margin_mse": float((mean[-query_count:] - margin_truth).square().mean()),
                    }
                )
    model.train()
    lineages = sorted({row["lineage"] for row in rows})
    summary = {
        metric: float(
            np.mean(
                [
                    np.mean([row[metric] for row in rows if row["lineage"] == lineage])
                    for lineage in lineages
                ]
            )
        )
        for metric in ("collision_nll", "brier", "margin_mse")
    }
    return summary, rows


def train(
    history, development, destination, *, uncorrelated=False, held_family=None, config=TRAINING
):
    if (
        not history
        or not development
        or any(a.role != "history" for a in history)
        or any(a.role != "development" for a in development)
        or len({a.source_id for a in (*history, *development)}) != len(history) + len(development)
    ):
        raise ValueError("Joint training requires disjoint history and development roles")
    if held_family and any(a.family == held_family for a in (*history, *development)):
        raise ValueError("Held family entered joint training or checkpoint selection")
    kinds = {a.scenarios[0].kind for a in (*history, *development)}
    if len(kinds) != 1:
        raise ValueError("Train one functional scenario at a time")
    specification = json.loads(
        json.dumps(
            {
                "kind": next(iter(kinds)),
                "uncorrelated": uncorrelated,
                "held_family": held_family,
                "config": config,
                "training_contract": training_contract(),
                "history_fingerprints": archive_fingerprints(history),
                "development_fingerprints": archive_fingerprints(development),
            }
        )
    )
    destination = Path(destination)
    report_path = destination.with_suffix(".json")
    if destination.exists() or report_path.exists():
        report = read_json(report_path)
        if report["status"] != "complete" or not specification_matches(
            report["specification"], specification
        ):
            raise ValueError("Existing joint training is incomplete or belongs to different inputs")
        load_model(destination)
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    generator = np.random.default_rng(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JointResponseKernel(history[0].x.shape[1], uncorrelated=uncorrelated).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    started = time.perf_counter()
    best, curve = (float("inf"), float("inf")), []
    write_json(
        report_path,
        {"status": "running", "specification": specification, "curve": [], "device": device},
    )
    for step in range(1, config["steps"] + 1):
        archive = history[int(generator.integers(len(history)))]
        count = int(generator.choice(config["support_counts"]))
        indices = generator.choice(len(archive.scenarios), count, replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss = (
            joint_loss(
                model, archive.x[indices], archive.collision[indices], archive.safety[indices, 0]
            )
            / count
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite joint task evidence at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % config["validation_interval"] == 0 or step == config["steps"]:
            summary, episodes = validate(model, development, config)
            score = summary["collision_nll"], summary["brier"]
            curve.append(
                {
                    "step": step,
                    "training_nll": float(loss.detach()),
                    "validation": summary,
                    "episodes": episodes,
                    "correlation": float(model.correlation.detach()),
                }
            )
            if score < best:
                best = score
                temporary = destination.with_suffix(".tmp")
                torch.save(
                    {
                        "format": "joint-collision-response",
                        "dimension": model.dimension,
                        "state_dict": model.state_dict(),
                        "specification": specification,
                        "step": step,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                    },
                    temporary,
                )
                temporary.replace(destination)
            write_json(
                report_path,
                {
                    "status": "complete" if step == config["steps"] else "running",
                    "specification": specification,
                    "curve": curve,
                    "best_score": list(best),
                    "device": device,
                    "wall_s": time.perf_counter() - started,
                },
            )
            print(
                "joint_training",
                next(iter(kinds)),
                "uncorrelated",
                uncorrelated,
                step,
                "/",
                config["steps"],
                "validation_nll",
                round(score[0], 6),
                flush=True,
            )
    return read_json(report_path)


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "joint-collision-response" or not source_contract_matches(
        payload["specification"]["training_contract"], training_contract()
    ):
        raise ValueError("Joint model format or training code differs")
    model = JointResponseKernel(
        payload["dimension"], uncorrelated=payload["specification"]["uncorrelated"]
    )
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload
