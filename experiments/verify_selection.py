"""Independently reconstruct source weights and next-scenario probabilities."""

import copy

import numpy as np
import torch

from testing_methods.common.gp_classifier import ClassificationKernel, fit_target

from .inference_reference import probabilities_numpy
from .protocol import ROOT
from .scenarios import features

OUTPUT = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/stage_a"

CHECKS = (9, 32, 64, 128)


def reconstruct(candidates, base, sources, rows, mode, initial, *, device="cpu"):
    """Rebuild each audited state from the initial weights and previously paid Booleans."""
    x = features(candidates)
    physical = ClassificationKernel(x.shape[1]).to(device)
    target = copy.deepcopy(base).to(device).double()
    feedback_enabled = mode.endswith("F1")
    indices = [row["index"] for row in rows] if feedback_enabled else []
    labels = [row["feedback"]["collision"] for row in rows] if feedback_enabled else []
    if indices:
        fit_target(target, x[indices], labels, steps=10, learning_rate=0.003)
        fit_target(physical, x[indices], labels, steps=10, learning_rate=0.01)
    with torch.no_grad():
        mean = (
            torch.tensor(sources["means"], dtype=torch.float64, device=device)
            + target.mean
            - base.mean.to(device)
        )
        covariance = (
            torch.tensor(sources["covariances"], dtype=torch.float64, device=device)
            + target.covariance(x)[None]
        )
        mean = torch.cat((mean, target.mean.expand(1, len(x)), physical.mean.expand(1, len(x))))
        covariance = torch.cat(
            (covariance, target.covariance(x)[None], physical.covariance(x)[None])
        )
        probabilities = torch.tensor(
            probabilities_numpy(mean.cpu().numpy(), covariance.cpu().numpy(), indices, labels),
            dtype=torch.float64,
        )
        log_weights = torch.tensor(np.r_[0.75 * sources["weights"], 0.25]).log()
        if feedback_enabled:
            for row in rows:
                p = torch.tensor(row["prequery_component_probabilities"][:-1], dtype=torch.float64)
                log_weights += (p if row["feedback"]["collision"] else 1 - p).clamp_min(1e-15).log()
        weights = torch.cat(
            (0.9 * log_weights.softmax(0), torch.tensor([0.1], dtype=torch.float64))
        )
        combined = (weights[:, None] * probabilities).sum(0).numpy()
    remaining = set(range(len(x))) - {row["index"] for row in rows}
    maximum = max(combined[i] for i in remaining)
    index = (
        initial[len(rows)]
        if len(rows) < len(initial)
        else min(
            (i for i in remaining if combined[i] >= maximum - 1e-12),
            key=lambda i: candidates[i].scenario_id,
        )
    )
    return index, combined, weights.numpy(), probabilities.numpy()
