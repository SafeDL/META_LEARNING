"""The final matrix, admitted models and outcome-independent protocol."""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
import torch

from controllers.highway.catalog import INDEPENDENT_SUTS
from controllers.highway.learning import CHECKPOINT_ROOT
from controllers.highway.train import TRAINING, training_contract
from testing_methods.baselines.indago import method as indago
from testing_methods.baselines.scenario_fuzz import model as scenario_fuzz
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.domain_training import DEFAULT as MODEL_CONFIG
from testing_methods.history_transfer.domain_training import load_model, specification

from .construction import load_construction
from .discovery_config import read_json, write_json
from .domain_comparison import REFERENCES, baseline_configuration
from .domain_comparison import source_contract as domain_source_contract
from .executions import execution_contract, require_qualified
from .independent_models import (
    FAMILIES,
    check_target_reuse,
    job_specification,
    load_sources,
    model_root,
    registered_source,
    scope_name,
)
from .independent_preparation import OUTPUT, validate_preparation
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT, runtime_environment
from .scenarios import DOMAINS, sample
from .source_registration import (
    maintained_path,
    source_contract_matches,
    specification_matches,
    validate_sources,
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


def source_contract():
    paths = set(domain_source_contract())
    paths.update(
        "experiments/" + name + ".py"
        for name in (
            "independent_protocol",
            "independent_preparation",
            "independent_models",
            "independent_data_audit",
            "independent_target_banks",
            "independent_validation",
            "independent_validation_audit",
            "independent_validation_report",
            "independent_workflow",
        )
    )
    paths.update(
        (
            "testing_methods/history_transfer/domain_training.py",
            "testing_methods/history_transfer/domain_consistency.py",
            "controllers/highway/catalog.py",
            "simulator/highway/interaction_env.py",
            "experiments/audit_collision_data.py",
            "experiments/execution_events.py",
            "experiments/source_registration.py",
            "experiments/statistics.py",
            "testing_methods/selection.py",
        )
    )
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)}


def construction_recipe():
    return json.loads(
        json.dumps(
            {
                "tasks": tasks(),
                "methods": list(METHODS),
                "controls": list(CONTROLS),
                "methods_by_condition": {
                    condition: list(methods_for(condition)) for condition in CONDITIONS
                },
                "control_scope": "all_sources; cut_in_interaction only",
                "matrix_runs": 1800,
                "additional_control_runs": 80,
                "independent_runs": len(jobs()),
                "reused_references": len(reuse_jobs()),
                "logical_comparison_rows": 2480,
                "reused_methods": list(REUSED_METHODS),
                "equivalent_queries": 240640,
                "candidate_count": 512,
                "budget": 128,
                "initial_count": 8,
                "checkpoints": [16, 32, 64, 128],
                "candidate_split": "new",
                "candidate_ids": {
                    kind: [s.scenario_id for s in sample(kind, 512, "new")] for kind in DOMAINS
                },
                "main": {
                    "covariance_mode": "diagonal",
                    "global_reliability": True,
                    "greedy": True,
                    "local_weights": False,
                    "lookahead": False,
                    "history_mass": 0.75,
                    "independent_target_mass": 0.25,
                    "physical_mass": 0.1,
                    "beta": 1.0,
                    "target_steps": 10,
                    "target_lr": 0.003,
                    "physical_steps": 10,
                    "physical_lr": 0.01,
                    "mode_gradient_tolerance": 1e-10,
                    "online_feature_precision": "float64",
                    "device": "cpu",
                    "historical_source_initialization": "uniform over allowed historical sources; independent of ADS family",
                },
                "main_training": asdict(MODEL_CONFIG),
                "baseline_configuration": baseline_configuration(),
                "indago_single_class_rule": "two-logit cross entropy; balanced weights over observed historical classes; missing-class weight zero",
                "source_files": source_contract(),
                "workers": WORKERS,
                "statistics": {
                    "aggregation": "equal lineage, then equal within-lineage runs",
                    "bootstrap_repetitions": 2000,
                    "bootstrap_seed": 1501,
                    "zero_collision_tasks": "retain count and AUC; recall not applicable",
                },
                "reference_set": list(REFERENCES),
                "development_selection_evidence": "domain_comparison/diagonal_controls/finalization.json",
            }
        )
    )


def model_artifacts():
    artifacts, scopes = {}, []
    original_files = registered_source()
    for name in ("registered_source.zip", "registered_source.json"):
        path = OUTPUT / "model_construction" / name
        artifacts[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for kind in DOMAINS:
        for held in FAMILIES:
            history, development = load_construction(kind, held)
            expected = specification(history, development, held, MODEL_CONFIG, "diagonal")
            main_root = model_root("gcrt", kind, held)
            paths = [
                main_root / "target_kernel.pt",
                main_root / "historical_encoder.pt",
                main_root / "source_priors.npz",
            ]
            for path, part in zip(paths[:2], ("target", "historical_encoder")):
                _, metadata = load_model(path)
                report = read_json(path.with_suffix(".json"))
                if (
                    metadata["part"] != part
                    or not specification_matches(metadata["specification"], expected)
                    or report["status"] != "complete"
                    or report["curve"][-1]["step"] != MODEL_CONFIG.steps
                ):
                    raise ValueError(
                        "A final main component was not completely constructed under its allowed scope"
                    )
                if part == "target":
                    reuse = metadata["reused_from"]
                    original = ROOT / maintained_path(reuse["checkpoint"])
                    if (
                        hashlib.sha256(original.read_bytes()).hexdigest()
                        != reuse["checkpoint_sha256"]
                        or hashlib.sha256(original.with_suffix(".json").read_bytes()).hexdigest()
                        != reuse["report_sha256"]
                    ):
                        raise ValueError("A reused target kernel's original evidence changed")
                    check_target_reuse(reuse["original_specification"], expected, original_files)
                    original_state = torch.load(original, map_location="cpu", weights_only=True)[
                        "state_dict"
                    ]
                    if any(
                        not torch.equal(value, original_state[name])
                        for name, value in metadata["state_dict"].items()
                    ):
                        raise ValueError(
                            "A reused target kernel does not retain its exact learned parameters"
                        )
                    for original_path in (original, original.with_suffix(".json")):
                        artifacts[str(original_path.relative_to(ROOT)).replace("\\", "/")] = (
                            hashlib.sha256(original_path.read_bytes()).hexdigest()
                        )
            sources = load_sources(kind, held)
            if (
                sources["names"] != [a.source_id for a in history]
                or sources["means"].shape != (len(history), 512)
                or not np.allclose(
                    sources["weights"], np.full(len(history), 1 / len(history)), atol=1e-15, rtol=0
                )
            ):
                raise ValueError(
                    "Final historical priors differ from the allowed sources and public pool"
                )
            prior = read_json(paths[2].with_suffix(".json"))
            if (
                prior["status"] != "complete"
                or prior["specification"]["candidate_ids"]
                != [s.scenario_id for s in sample(kind, 512, "new")]
                or prior["specification"]["encoder_sha256"]
                != hashlib.sha256(paths[1].read_bytes()).hexdigest()
                or prior["specification"]["history_fingerprints"] != archive_fingerprints(history)
                or prior["specification"]["held_family"] != held
            ):
                raise ValueError(
                    "Final historical priors belong to another independent candidate library"
                )
            for owner, module, filename in (
                ("indago", indago, "classifier.pt"),
                ("scenario_fuzz", scenario_fuzz, "risk.pt"),
            ):
                if owner == "indago" and held is not None:
                    continue
                classifier_job = read_json(
                    OUTPUT
                    / "model_construction/required_classifiers"
                    / owner
                    / kind
                    / f"{scope_name(held)}.json"
                )
                if classifier_job["status"] != "complete" or not specification_matches(
                    classifier_job["specification"], job_specification(owner, kind, held, history)
                ):
                    raise ValueError(
                        "The final classifier construction has not completed under its allowed source"
                    )
                path = model_root(owner, kind, held) / filename
                _, metadata = module.load(path)
                if (
                    metadata["history_fingerprints"] != archive_fingerprints(history)
                    or metadata["history_ids"] != [a.source_id for a in history]
                    or metadata["held_family"] != held
                    or metadata["config"] != asdict(module.DEFAULT)
                    or len(metadata["curve"]) != (module.DEFAULT.epochs)
                ):
                    raise ValueError(
                        "A final baseline classifier contains a disallowed history or construction"
                    )
                paths.append(path)
            artifacts.update(
                {
                    str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in paths
                }
            )
            scopes.append(
                {
                    "scenario": kind,
                    "scope": scope_name(held),
                    "history": archive_fingerprints(history),
                    "development": archive_fingerprints(development),
                }
            )
    return artifacts, scopes


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


def freeze():
    if (OUTPUT / "protocol.json").exists():
        return validate_protocol()
    validate_preparation()
    require_independent_qualification()
    banks = read_json(OUTPUT / "target_bank_index.json")
    bank_protocol = read_json(OUTPUT / "target_bank_protocol.json")
    current_contracts = {
        s.sut_id: {kind: execution_contract(s, kind=kind) for kind in DOMAINS}
        for s in INDEPENDENT_SUTS
    }
    if (
        bank_protocol["objects"] != json.loads(json.dumps([asdict(s) for s in INDEPENDENT_SUTS]))
        or bank_protocol["contracts"] != json.loads(json.dumps(current_contracts))
        or bank_protocol["expected_physical_calls"] != 20480
        or bank_protocol["collector_sha256"]
        != hashlib.sha256(
            (ROOT / "experiments/independent_target_banks.py").read_bytes()
        ).hexdigest()
        or banks["status"] != "complete"
        or len(banks["banks"]) != 40
    ):
        raise ValueError("Complete the fixed independent raw pools before final selector freezing")
    files = registered_source()
    if any(
        hashlib.sha256(files[name]).hexdigest() != digest
        for name, digest in bank_protocol["model_and_selection_source_files"].items()
    ):
        raise ValueError("The original pre-outcome method registration cannot be verified")
    recipe = construction_recipe()
    for name in (
        "candidate_count",
        "budget",
        "initial_count",
        "candidate_split",
        "candidate_ids",
        "main_training",
        "baseline_configuration",
        "statistics",
    ):
        if bank_protocol["recipe"][name] != recipe[name]:
            raise ValueError(
                "An unapproved physical pool, training budget or statistical setting changed"
            )
    audit = read_json(OUTPUT / "construction_data_audit.json")
    if audit["status"] != "passed" or audit["checked_executions"] != 30720:
        raise ValueError(
            "Audit all final allowed historical/development banks before independent selection"
        )
    regression_path = OUTPUT / "reduced_regression.xml"
    regression = ET.parse(regression_path).getroot()
    suites = list(regression.iter("testsuite"))
    if (
        not suites
        or sum(int(s.attrib.get("tests", 0)) for s in suites) == 0
        or any(int(s.attrib.get(k, 0)) for s in suites for k in ("failures", "errors"))
    ):
        raise ValueError("The final complete engineering regression has not passed")
    supplemental = OUTPUT / "reduced_integration_regression.xml"
    supplemental_suites = list(ET.parse(supplemental).getroot().iter("testsuite"))
    if (
        not supplemental_suites
        or sum(int(s.attrib.get("tests", 0)) for s in supplemental_suites) == 0
        or any(int(s.attrib.get(k, 0)) for s in supplemental_suites for k in ("failures", "errors"))
    ):
        raise ValueError("The reduced report integration and zero-collision checks have not passed")
    artifacts, scopes = model_artifacts()
    protocol = {
        "recipe": construction_recipe(),
        "models": artifacts,
        "scopes": scopes,
        "objects": [asdict(s) for s in INDEPENDENT_SUTS],
        "sut_contracts": {
            s.sut_id: {kind: execution_contract(s, kind=kind) for kind in DOMAINS}
            for s in INDEPENDENT_SUTS
        },
        "runtime_environment": runtime_environment(),
        "history_audit": audit["status"],
        "regression_file": str(regression_path.relative_to(ROOT)).replace("\\", "/"),
        "regression_sha256": hashlib.sha256(regression_path.read_bytes()).hexdigest(),
        "supplemental_regressions": {
            str(supplemental.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                supplemental.read_bytes()
            ).hexdigest()
        },
        "target_bank_protocol_sha256": hashlib.sha256(
            (OUTPUT / "target_bank_protocol.json").read_bytes()
        ).hexdigest(),
        "target_bank_index_sha256": hashlib.sha256(
            (OUTPUT / "target_bank_index.json").read_bytes()
        ).hexdigest(),
        "approved_registration_changes": {
            "basis": "user-approved design in docs/collision_discovery_plan.md section 8",
            "matrix": "1880 unique runs and 600 source-agnostic reuse references",
            "source_initialization": "equal historical sources; same rule in new encoder training and inference",
            "execution_schedule": "16 independent CPU runs; one native thread per selector; unchanged method budgets",
            "original_objects_candidates_physics_and_feedback": "unchanged; retain original raw-pool registration",
            "original_method_source_snapshot": "model_construction/registered_source.zip",
        },
        "independent_selection_results_read_before_freezing": False,
    }
    protocol = json.loads(json.dumps(protocol))
    path = OUTPUT / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("The independent validation was already frozen differently")
    write_json(path, protocol)
    return protocol


def validate_protocol():
    protocol = read_json(OUTPUT / "protocol.json")
    expected = dict(protocol["recipe"])
    current = construction_recipe()
    validate_sources(expected.pop("source_files"), current.pop("source_files"))
    if expected != current:
        raise ValueError("The independent method, task list or evaluation changed after freezing")
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
