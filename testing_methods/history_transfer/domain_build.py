"""Build the current method from allowed historical archives, without old snapshots."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.construction import load_construction
from experiments.discovery_config import read_json, write_json
from experiments.protocol import ROOT
from experiments.scenarios import features
from testing_methods.common.fsbo.method import CONFIG as TARGET_CONFIG
from testing_methods.common.fsbo.method import load as load_target
from testing_methods.common.fsbo.method import train as train_target
from testing_methods.common.gp_laplace import posterior as binary_posterior
from testing_methods.common.response import archive_fingerprints, source_weights

from .joint_response import MARGIN_NOISE, margin_values
from .joint_training import load_model
from .joint_training import train as train_response

OUTPUT = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/models"


def historical_posterior(model, archive, query_x):
    """Condition on historical safety, then converge the historical Bernoulli posterior."""
    count = len(archive.x)
    x = np.concatenate((archive.x, query_x))
    size = len(x)
    with torch.no_grad():
        mean = model.joint_mean(size)
        covariance = model.joint_covariance(x)
        margins = torch.arange(size, size + count, device=mean.device)
        cross = covariance[:, margins]
        gram = covariance[margins][:, margins]
        factor = torch.linalg.cholesky(
            gram + MARGIN_NOISE**2 * torch.eye(count, dtype=mean.dtype, device=mean.device)
        )
        residual = (
            torch.as_tensor(
                margin_values(archive.safety[:, 0]), dtype=mean.dtype, device=mean.device
            )
            - mean[margins]
        )
        mean = mean + cross @ torch.cholesky_solve(residual[:, None], factor)[:, 0]
        solved = torch.linalg.solve_triangular(factor, cross.T, upper=False)
        covariance = covariance - solved.T @ solved
        mean, covariance, diagnostics = binary_posterior(
            mean[:size], covariance[:size, :size], list(range(count)), archive.collision
        )
        mean, covariance = mean[count:], covariance[count:, count:]
        torch.linalg.cholesky(
            covariance + 1e-7 * torch.eye(len(query_x), dtype=mean.dtype, device=mean.device)
        )
    return mean.cpu().numpy(), covariance.cpu().numpy(), diagnostics


def build_sources(kind, history, candidates, checkpoint, destination):
    model, _ = load_model(checkpoint)
    specification = {
        "scenario": kind,
        "history_fingerprints": archive_fingerprints(history),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "candidate_ids": [s.scenario_id for s in candidates],
        "historical_posterior_solver": "converged Bernoulli/Laplace; gradient at most 1e-10",
    }
    destination = Path(destination)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if json.loads(str(saved["specification"])) != specification:
                raise ValueError(
                    "Historical priors belong to different allowed responses or public candidates"
                )
        return read_json(destination.with_suffix(".json"))
    started = time.perf_counter()
    means, covariances, diagnostics = [], [], []
    for archive in history:
        mean, covariance, diagnostic = historical_posterior(model, archive, features(candidates))
        means.append(mean)
        covariances.append(covariance)
        diagnostics.append({"source_id": archive.source_id, **diagnostic})
        print("domain_historical_prior", kind, archive.source_id, flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            specification=np.asarray(json.dumps(specification)),
            means=np.stack(means),
            covariances=np.stack(covariances),
            weights=source_weights(history),
            names=np.asarray([a.source_id for a in history]),
        )
    report = {
        "status": "complete",
        "specification": specification,
        "diagnostics": diagnostics,
        "construction_s": time.perf_counter() - started,
        "new_physical_calls": 0,
    }
    write_json(destination.with_suffix(".json"), report)
    return report


def load_sources(kind):
    with np.load(OUTPUT / kind / "source_priors.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in ("means", "covariances", "weights")} | {
            "names": saved["names"].tolist()
        }


def main():
    torch.set_num_threads(1)
    reports = []
    with threadpool_limits(1):
        for kind in ("cut_in", "cut_out"):
            history, development = load_construction(kind)
            target_path = OUTPUT / kind / "target_kernel.pt"
            if target_path.exists():
                _, payload = load_target(target_path)
                if (
                    payload["history_fingerprints"] != archive_fingerprints(history)
                    or payload["development_fingerprints"] != archive_fingerprints(development)
                    or payload["config"] != TARGET_CONFIG
                ):
                    raise ValueError(
                        "Target kernel differs from the allowed historical construction"
                    )
                target_report = read_json(target_path.with_suffix(".json"))
            else:
                target_report = train_target(history, development, target_path)
            response_path = OUTPUT / kind / "history_response.pt"
            response_report = train_response(history, development, response_path)
            candidates = development[0].scenarios
            if any(a.scenarios != candidates for a in development):
                raise ValueError(
                    "Matched development objects must share the public candidate library"
                )
            source_report = build_sources(
                kind, history, candidates, response_path, OUTPUT / kind / "source_priors.npz"
            )
            reports.append(
                {
                    "scenario": kind,
                    "target_kernel": target_report,
                    "historical_response_encoder": response_report,
                    "source_priors": source_report,
                }
            )
            write_json(
                OUTPUT / "construction.json",
                {
                    "status": "complete" if len(reports) == 2 else "running",
                    "tasks": reports,
                    "new_physical_calls": 0,
                    "sequence": [
                        "historical collision deep-kernel learning",
                        "historical paired-response encoder learning",
                        "converged historical source posteriors",
                        "online Boolean target/physical adaptation",
                    ],
                    "target_kernel_foundation": "FSBO component; not a new FSBO comparison baseline",
                },
            )
    return reports


if __name__ == "__main__":
    main()
