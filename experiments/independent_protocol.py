"""The final matrix, admitted models and outcome-independent protocol."""

import hashlib
import zipfile
from dataclasses import asdict

from controllers.highway.catalog import INDEPENDENT_SUTS
from controllers.highway.learning import CHECKPOINT_ROOT
from controllers.highway.train import TRAINING, training_contract

from .development import REFERENCES
from .discovery_config import read_json
from .executions import require_qualified
from .independent_preparation import OUTPUT
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .scenarios import DOMAINS
from .source_registration import (
    maintained_path,
    source_contract_matches,
)

METHODS = ("gcrt", *REFERENCES)

CONTROLS = ("H1F0", "H0F1")

CONDITIONS = ("all_sources", "strict_leave_family")

REUSED_METHODS = ("random", "av_bo_rf_ei", "samota")

STRICT_METHODS = ("gcrt", "scenario_fuzz", *REUSED_METHODS)

WORKERS = 16


def tasks():
    return [
        {
            "target_id": sut.sut_id,
            "family": sut.family,
            "lineage": sut.lineage,
            "scenario": kind,
            "condition": condition,
            "environment_seed": 81001,
            "algorithm_seed": seed,
        }
        for sut in INDEPENDENT_SUTS
        for kind in DOMAINS
        for condition in CONDITIONS
        for seed in DEFAULT.algorithm_seeds
    ]


def jobs():
    return [
        (task, method)
        for task in tasks()
        for method in (
            (*METHODS, *CONTROLS)
            if task["condition"] == "all_sources"
            else ("gcrt", "scenario_fuzz")
        )
        if method not in CONTROLS or task["scenario"] == "cut_in_interaction"
    ]


def reuse_jobs():
    return [
        (task, method)
        for task in tasks()
        if task["condition"] == "strict_leave_family"
        for method in REUSED_METHODS
    ]


def methods_for(condition):
    return METHODS if condition == "all_sources" else STRICT_METHODS


def canonical_task(task, method):
    if task["condition"] == "strict_leave_family" and method in REUSED_METHODS:
        return {**task, "condition": "all_sources"}
    return task


def held_family(task):
    return task["family"] if task["condition"] == "strict_leave_family" else None


def target_bank_path(task):
    return (
        TEST_RECORDS_ROOT
        / "independent_targets"
        / task["target_id"]
        / task["scenario"]
        / "env_81001"
    )


def run_path(task, method):
    task = canonical_task(task, method)
    owner = "gcrt" if method in ("gcrt", *CONTROLS) else method
    return (
        ROOT
        / "experiment_outputs"
        / owner
        / "collision/discovery/independent_validation"
        / method
        / task["target_id"]
        / task["scenario"]
        / task["condition"]
        / f"{task['algorithm_seed']}.json"
    )


def require_independent_qualification():
    for sut in INDEPENDENT_SUTS:
        require_qualified(sut)
        if sut.family == "learning":
            report = read_json(CHECKPOINT_ROOT / sut.training_run / "training.json")
            if (
                report["configuration"] != TRAINING
                or report["actual_timesteps"] < TRAINING["total_timesteps"]
                or report["object"] != asdict(sut)
                or not source_contract_matches(report["source_contract"], training_contract())
            ):
                raise ValueError(
                    "An independent policy did not complete the fixed training budget and object contract"
                )


def validate_protocol():
    protocol = read_json(OUTPUT / "protocol.json")
    with zipfile.ZipFile(OUTPUT / "model_construction/frozen_sources.zip") as archive:
        sources = protocol["recipe"]["source_files"]
        if set(archive.namelist()) != set(sources):
            raise ValueError("The preserved source archive contains different files")
        for name, digest in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"A preserved source file changed: {name}")
    if (
        hashlib.sha256(
            (ROOT / maintained_path(protocol["regression_file"])).read_bytes()
        ).hexdigest()
        != protocol["regression_sha256"]
    ):
        raise ValueError("The independent engineering regression record changed after freezing")
    for name, expected in protocol["supplemental_regressions"].items():
        if hashlib.sha256((ROOT / maintained_path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError("A reduced-report regression record changed after freezing")
    for name in ("target_bank_protocol", "target_bank_index"):
        if (
            hashlib.sha256((OUTPUT / f"{name}.json").read_bytes()).hexdigest()
            != protocol[f"{name}_sha256"]
        ):
            raise ValueError(
                "The independent raw-pool construction changed after selector freezing"
            )
    for name, expected in protocol["models"].items():
        if hashlib.sha256((ROOT / maintained_path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError(f"A final historical model changed after freezing: {name}")
    return protocol


def construction_recipe():
    return read_json(OUTPUT / "protocol.json")["recipe"]
