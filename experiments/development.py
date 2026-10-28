"""Read preserved development tasks, models and selector inputs."""

import hashlib

from testing_methods.baselines.av_bo_rf_ei import method as rf_ei
from testing_methods.baselines.indago import method as indago_model
from testing_methods.baselines.indago import pool as indago
from testing_methods.baselines.paired_system_active_evaluation import method as paired
from testing_methods.baselines.response_baselines.collision import run as run_random
from testing_methods.baselines.samota import pool as samota
from testing_methods.baselines.scenario_fuzz import model as graph_model
from testing_methods.baselines.scenario_fuzz import pool as scenario_fuzz
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.selection import aggregate_history, select

from .discovery_config import read_json
from .domain_paired_validation import build_proxy
from .initialization import initial_indices
from .protocol import ROOT
from .recorded_models import DEVELOPMENT_MODELS as MODELS
from .recorded_models import load_development_sources as load_sources
from .recorded_models import load_model as load_target
from .source_registration import maintained_path

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/discovery/domain_comparison"

REFERENCES = (
    "random",
    "av_bo_rf_ei",
    "samota",
    "scenario_fuzz",
    "paired_system_active_evaluation",
    "indago",
)

METHODS = ("gcrt", *REFERENCES, "H0F0", "H1F0", "H0F1")

WORKERS = 8


def run_path(task, method):
    owner = "gcrt" if method in ("gcrt", "H0F0", "H1F0", "H0F1") else method
    return (
        ROOT
        / "experiment_outputs"
        / owner
        / "collision/discovery/domain_comparison/runs"
        / method
        / task["target_id"]
        / task["scenario"]
        / f"{task['algorithm_seed']}.json"
    )


def model_paths(kind):
    return (
        MODELS / kind / "target_kernel.pt",
        MODELS / kind / "history_response.pt",
        MODELS / kind / "source_priors.npz",
        ROOT
        / f"experiment_outputs/indago/collision/discovery/domain_comparison/stage_a/models/{kind}/classifier.pt",
        ROOT
        / f"experiment_outputs/scenario_fuzz/collision/discovery/domain_comparison/stage_a/models/{kind}/risk.pt",
    )


def baseline_configuration():
    return read_json(OUTPUT / "protocol.json")["baseline_config"]


def validate_protocol(specification):
    for path, digest in {**specification["models"], **specification["stage_a_evidence"]}.items():
        if hashlib.sha256((ROOT / maintained_path(path)).read_bytes()).hexdigest() != digest:
            raise ValueError("A preserved development artifact changed")


def public_task(task, method, candidates, history):
    public = {
        "candidates": candidates,
        "initial_indices": initial_indices(512, 8, task["algorithm_seed"]),
        "budget": 128,
    }
    if method in ("gcrt", "H0F0", "H1F0", "H0F1"):
        base, _ = load_target(MODELS / task["scenario"] / "target_kernel.pt")
        sources = load_sources(task["scenario"])
        if method.startswith("H0"):
            sources = aggregate_history(sources)
        return select, {
            **public,
            "pretrained": base,
            "sources": sources,
            "mode": "H1F1" if method == "gcrt" else method,
            "device": "cuda",
        }
    public["seed"] = task["algorithm_seed"]
    if method == "random":
        return run_random, {**public, "method": "random"}
    if method == "av_bo_rf_ei":
        return rf_ei.run, public
    if method == "samota":
        return samota.run, public
    if method == "paired_system_active_evaluation":
        proxy, _ = build_proxy(candidates, history)
        return paired.run, {**public, "proxy": proxy}
    model_path = model_paths(task["scenario"])[3 if method == "indago" else 4]
    loader, selector = (
        (indago_model.load, indago.run)
        if method == "indago"
        else (graph_model.load, scenario_fuzz.run)
    )
    model, metadata = loader(model_path)
    if metadata["history_fingerprints"] != archive_fingerprints(history):
        raise ValueError("A baseline classifier differs from the allowed historical responses")
    return selector, {**public, "classifier": model}
