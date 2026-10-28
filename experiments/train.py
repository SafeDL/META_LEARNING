"""Build the RSS method from recorded traces, then export fresh historical priors."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from testing_methods.common.response import archive_fingerprints, source_weights
from testing_methods.history_transfer.inference import historical_prior
from testing_methods.history_transfer.training import (
    COVARIANCE_MODE,
    HELD_FAMILIES,
    MODEL_ROOT,
    load_model,
    model_directory,
)
from testing_methods.history_transfer.training import main as train_models

from .construction import build_responses, load_training_data
from .discovery_config import write_json
from .protocol import DEFAULT, ROOT
from .rss import measurement_contract
from .scenarios import DOMAINS, features, sample


def source_predictions(model, history, candidates):
    means, variances = [], []
    query_x = features(candidates)
    with torch.no_grad():
        for archive in history:
            mean, covariance = historical_prior(model, archive, query_x)
            means.append(mean.cpu().numpy())
            variances.append(covariance.diagonal().cpu().numpy())
    return {
        "means": np.stack(means),
        "variances": np.stack(variances),
        "weights": source_weights(history),
        "names": [a.source_id for a in history],
    }


def build_priors(kind, held_family):
    torch.set_num_threads(1)
    root = model_directory(kind, held_family)
    history, _ = load_training_data(kind, held_family)
    model, metadata = load_model(root / "historical_encoder.pt")
    if (
        metadata["specification"]["history_fingerprints"] != archive_fingerprints(history)
        or metadata["specification"]["held_family"] != held_family
    ):
        raise ValueError("RSS encoder was trained on another history scope")
    candidates = sample(kind, DEFAULT.candidate_count, "new")
    spec = {
        "continuous_response": measurement_contract(),
        "history_fingerprints": archive_fingerprints(history),
        "model_specification": metadata["specification"],
        "candidate_ids": [s.scenario_id for s in candidates],
        "covariance_mode": COVARIANCE_MODE,
        "encoder_sha256": hashlib.sha256((root / "historical_encoder.pt").read_bytes()).hexdigest(),
    }
    path = root / "source_priors.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["specification"])) != spec:
                raise ValueError("RSS priors use different responses or public coordinates")
    else:
        sources = source_predictions(model, history, candidates)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                means=sources["means"],
                variances=sources["variances"],
                weights=sources["weights"],
                names=np.asarray(sources["names"]),
                specification=json.dumps(spec),
            )
    return {"scenario": kind, "scope": root.name, "path": str(path.relative_to(ROOT))}


def load_sources(kind, held_family=None):
    root = model_directory(kind, held_family)
    with np.load(root / "source_priors.npz", allow_pickle=False) as saved:
        specification = json.loads(str(saved["specification"]))
        if (
            specification["continuous_response"] != measurement_contract()
            or specification["encoder_sha256"]
            != hashlib.sha256((root / "historical_encoder.pt").read_bytes()).hexdigest()
        ):
            raise ValueError("RSS sources do not match the current global measure and encoder")
        return {
            "means": saved["means"].copy(),
            "covariances": np.stack([np.diag(v) for v in saved["variances"]]),
            "weights": saved["weights"].copy(),
            "names": saved["names"].tolist(),
        }


def main():
    build_responses()
    train_models()
    reports = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(build_priors, kind, held) for kind in DOMAINS for held in HELD_FAMILIES
        ]
        for future in as_completed(futures):
            reports.append(future.result())
            print(f"RSS historical priors {len(reports)}/25", flush=True)
    result = {
        "status": "complete",
        "continuous_response": measurement_contract(),
        "priors": reports,
        "target_feedback": "collision Boolean only",
        "new_simulator_calls": 0,
        "independent_results": "Old clearance-based runs are not RSS validation results.",
    }
    write_json(MODEL_ROOT / "workflow.json", result)
    return result


if __name__ == "__main__":
    main()
