"""Read the preserved clearance-study models and priors; no model construction."""

from pathlib import Path

import numpy as np
import torch

from testing_methods.common.gp_classifier import ClassificationKernel
from testing_methods.history_transfer.model import JointResponseKernel

from .protocol import ROOT

DEVELOPMENT_MODELS = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/models"
FAMILIES = (None, "rule", "optimization", "search", "learning")


def scope_name(held_family):
    return "all_sources" if held_family is None else f"without_{held_family}"


def model_root(owner, kind, held_family):
    directory = "uniform_models" if owner == "gcrt" else "independent_models"
    return (
        ROOT
        / "experiment_outputs"
        / owner
        / "collision"
        / directory
        / kind
        / scope_name(held_family)
    )


def load_sources(kind, held_family):
    path = model_root("gcrt", kind, held_family) / "source_priors.npz"
    with np.load(path, allow_pickle=False) as saved:
        return {
            "means": saved["means"].copy(),
            "covariances": np.stack([np.diag(v) for v in saved["variances"]]),
            "weights": saved["weights"].copy(),
            "names": saved["names"].tolist(),
        }


def load_development_sources(kind):
    with np.load(DEVELOPMENT_MODELS / kind / "source_priors.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in ("means", "covariances", "weights")} | {
            "names": saved["names"].tolist()
        }


def load_model(path):
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved["format"] == "domain-consistent-collision":
        model = (
            ClassificationKernel(saved["dimension"], deep=True)
            if saved["part"] == "target"
            else JointResponseKernel(saved["dimension"])
        )
        model = model.double()
    elif saved["format"] == "joint-collision-response":
        model = JointResponseKernel(saved["dimension"])
    elif saved["format"] == "fsbo-collision":
        model = ClassificationKernel(saved["dimension"], deep=True)
    else:
        raise ValueError("Unknown recorded model format")
    model.load_state_dict(saved["state_dict"])
    if saved["format"] == "joint-collision-response" and saved["specification"]["uncorrelated"]:
        with torch.no_grad():
            model.raw_correlation.zero_()
    return model.eval(), saved
