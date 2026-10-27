"""Construct only the historical models needed by the three independent experiments."""

import argparse
import ast
import hashlib
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from testing_methods.baselines.indago import method as indago
from testing_methods.baselines.scenario_fuzz import model as scenario_fuzz
from testing_methods.common.response import archive_fingerprints, source_weights
from testing_methods.history_transfer.domain_consistency import historical_prior
from testing_methods.history_transfer.domain_training import (
    DEFAULT,
    build_scope,
    load_model,
    specification,
)

from .construction import load_construction
from .discovery_config import read_json, write_json
from .independent_preparation import OUTPUT, validate_preparation
from .protocol import ROOT
from .scenarios import DOMAINS, features, sample
from .source_registration import maintained_path, source_contract_matches, specification_matches

FAMILIES = (None, "rule", "optimization", "search", "learning")
WORKERS = 8


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


def registered_source():
    path = OUTPUT / "model_construction/registered_source.zip"
    index = read_json(path.with_suffix(".json"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != index["archive_sha256"]:
        raise ValueError("The original registered construction source changed")
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in index["source_sha256"]}
    if any(
        hashlib.sha256(files[name]).hexdigest() != digest
        for name, digest in index["source_sha256"].items()
    ):
        raise ValueError("A registered construction source file changed")
    return files


def check_target_reuse(old_specification, expected, files):
    """Prove the reused target's numerical training is unaffected by source weighting."""
    for key, value in expected.items():
        if key != "training_contract" and old_specification[key] != value:
            raise ValueError("A reused target kernel has another data scope or configuration")
    for role in ("history_fingerprints", "development_fingerprints"):
        if list(old_specification[role]) != list(expected[role]):
            raise ValueError(
                "The reused target kernel's training or validation source order changed"
            )
    changed = {
        "testing_methods/history_transfer/domain_training.py",
        "testing_methods/common/response.py",
    }
    for name, digest in old_specification["training_contract"].items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("The archived source does not describe the original target kernel")
        current_name = maintained_path(name)
        current_digest = hashlib.sha256((ROOT / current_name).read_bytes()).hexdigest()
        if current_name not in changed and not source_contract_matches(
            {name: digest}, {current_name: current_digest}
        ):
            raise ValueError("A numerical dependency of the original target kernel changed")
    nodes = {
        "testing_methods/history_transfer/domain_training.py": (
            "Config",
            "selected_indices",
            "target_validation",
            "save_model",
            "train_target",
        ),
        "testing_methods/common/response.py": ("ResponseArchive", "archive_fingerprints"),
    }
    for name, required in nodes.items():
        original_name = next(key for key in files if maintained_path(key) == name)
        original = {
            node.name: numerical_training_ast(node)
            for node in ast.parse(files[original_name]).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        current = {
            node.name: numerical_training_ast(node)
            for node in ast.parse((ROOT / name).read_bytes()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        if any(original[node] != current[node] for node in required):
            raise ValueError(
                "The original target-kernel training or validation computation changed"
            )


def numerical_training_ast(node):
    # The completed-checkpoint branch checks metadata; it does not train parameters.
    if node.name == "train_target":
        node.body = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.If)
                and any(
                    isinstance(part, ast.Attribute) and part.attr == "exists"
                    for part in ast.walk(statement.test)
                )
            )
        ]
    return ast.dump(node)


def reuse_target(kind, held_family, history, development):
    destination = model_root("gcrt", kind, held_family) / "target_kernel.pt"
    expected = specification(history, development, held_family, DEFAULT, "diagonal")
    if destination.exists():
        _, metadata = load_model(destination)
        if not specification_matches(metadata["specification"], expected):
            raise ValueError("The reused target kernel has another current construction scope")
        return read_json(destination.with_suffix(".json"))
    original = (
        ROOT
        / "experiment_outputs/gcrt/collision/consistent_models"
        / kind
        / scope_name(held_family)
        / "target_kernel.pt"
    )
    saved = torch.load(original, map_location="cpu", weights_only=True)
    report = read_json(original.with_suffix(".json"))
    if (
        saved["part"] != "target"
        or saved["format"] != "domain-consistent-collision"
        or report["status"] != "complete"
        or report["curve"][-1]["step"] != DEFAULT.steps
        or report["specification"] != saved["specification"]
    ):
        raise ValueError("Only a completed, unchanged historical target kernel may be reused")
    check_target_reuse(saved["specification"], expected, registered_source())
    evidence = {
        "checkpoint": str(original.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(original.with_suffix(".json").read_bytes()).hexdigest(),
        "original_specification": saved["specification"],
        "verification": "unchanged data/configuration, numerical dependency hashes and target-training AST; exact retained state_dict",
        "new_training_s": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**saved, "specification": expected, "reused_from": evidence}, destination)
    report = {**report, "specification": expected, "reused_from": evidence, "new_training_s": 0}
    write_json(destination.with_suffix(".json"), report)
    return report


def construct_main(kind, held_family):
    history, development = load_construction(kind, held_family)
    reuse_target(kind, held_family, history, development)
    return build_scope(kind, held_family, "diagonal")


def job_specification(owner, kind, held_family, history):
    module = indago if owner == "indago" else scenario_fuzz
    paths = [
        f"testing_methods/baselines/{owner}/" + name
        for name in (("method.py",) if owner == "indago" else ("model.py", "graphs.py"))
    ]
    return {
        "owner": owner,
        "scenario": kind,
        "held_family": held_family,
        "history": archive_fingerprints(history),
        "config": asdict(module.DEFAULT),
        "source_files": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths
        },
    }


def construct_classifier(owner, kind, held_family):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, _ = load_construction(kind, held_family)
        spec = job_specification(owner, kind, held_family, history)
        path = (
            OUTPUT
            / "model_construction/required_classifiers"
            / owner
            / kind
            / f"{scope_name(held_family)}.json"
        )
        module, filename = (
            (indago, "classifier.pt") if owner == "indago" else (scenario_fuzz, "risk.pt")
        )
        destination = model_root(owner, kind, held_family) / filename
        started = time.perf_counter()
        previous = read_json(path) if path.exists() else None
        completed = previous is not None and previous["status"] == "complete"
        if completed and (
            not specification_matches(previous["specification"], spec) or not destination.exists()
        ):
            raise ValueError(
                "A completed classifier record differs from its required model or source"
            )
        if not completed:
            write_json(path, {"status": "running", "specification": spec})
        reused = destination.exists()
        if reused:
            files = registered_source()
            original_contract = {
                name: hashlib.sha256(content).hexdigest()
                for name, content in files.items()
                if maintained_path(name) in spec["source_files"]
            }
            if not source_contract_matches(original_contract, spec["source_files"]):
                raise ValueError("A reused classifier's training implementation changed")
            _, report = module.load(destination)
            report = {k: v for k, v in report.items() if k != "state_dict"}
        else:
            report = module.train(history, destination, held_family=held_family)
        if (
            report["history_fingerprints"] != spec["history"]
            or report["history_ids"] != [a.source_id for a in history]
            or report["held_family"] != held_family
            or report["config"] != spec["config"]
            or len(report["curve"]) != module.DEFAULT.epochs
        ):
            raise ValueError("A classifier has not completed its exact allowed historical training")
        if not completed:
            write_json(
                path,
                {
                    "status": "complete",
                    "specification": spec,
                    "report": report,
                    "reused": reused,
                    "new_training_s": 0 if reused else report["training_s"],
                    "elapsed_s": time.perf_counter() - started,
                },
            )
        else:
            reused = previous["reused"]
        return {
            "owner": owner,
            "scenario": kind,
            "scope": scope_name(held_family),
            "manifest": str(path.relative_to(ROOT)),
            "reused": reused,
        }


def construct_priors(kind, held_family):
    torch.set_num_threads(1)
    with threadpool_limits(1), torch.no_grad():
        history, development = load_construction(kind, held_family)
        root = model_root("gcrt", kind, held_family)
        model, metadata = load_model(root / "historical_encoder.pt")
        candidates = sample(kind, 512, "new")
        spec = {
            "scenario": kind,
            "held_family": held_family,
            "history_fingerprints": archive_fingerprints(history),
            "encoder_sha256": hashlib.sha256(
                (root / "historical_encoder.pt").read_bytes()
            ).hexdigest(),
            "candidate_ids": [s.scenario_id for s in candidates],
            "covariance_mode": "diagonal",
        }
        if not specification_matches(
            metadata["specification"],
            specification(history, development, held_family, DEFAULT, "diagonal"),
        ):
            raise ValueError(
                "The final response encoder differs from the allowed construction scope"
            )
        path = root / "source_priors.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved["specification"])) != spec:
                    raise ValueError(
                        "Source priors differ from their frozen public candidate coordinates"
                    )
            return read_json(path.with_suffix(".json"))
        started = time.perf_counter()
        means, variances = [], []
        for archive in history:
            mean, covariance = historical_prior(model, archive, features(candidates))
            if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
                raise FloatingPointError(
                    "Nonfinite historical posterior for independent candidates"
                )
            torch.linalg.cholesky(covariance + 1e-7 * torch.eye(512, dtype=covariance.dtype))
            means.append(mean.cpu().numpy())
            variances.append(covariance.diagonal().cpu().numpy())
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                means=np.stack(means),
                variances=np.stack(variances),
                weights=source_weights(history),
                names=np.asarray([a.source_id for a in history]),
                specification=np.asarray(json.dumps(spec)),
            )
        report = {
            "status": "complete",
            "specification": spec,
            "construction_s": time.perf_counter() - started,
            "new_physical_calls": 0,
        }
        write_json(path.with_suffix(".json"), report)
        return report


def load_sources(kind, held_family):
    path = model_root("gcrt", kind, held_family) / "source_priors.npz"
    with np.load(path, allow_pickle=False) as saved:
        return {
            "means": saved["means"].copy(),
            "covariances": np.stack([np.diag(v) for v in saved["variances"]]),
            "weights": saved["weights"].copy(),
            "names": saved["names"].tolist(),
        }


def available_jobs(stage):
    if stage == "classifiers":
        return [
            (owner, kind, held)
            for kind in DOMAINS
            for owner in ("scenario_fuzz", "indago")
            for held in (FAMILIES if owner == "scenario_fuzz" else (None,))
        ]
    return [(kind, held) for kind in DOMAINS for held in FAMILIES]


def build_job(stage, arguments):
    name = "_".join(str(arg) if arg is not None else "all_sources" for arg in arguments)
    path = OUTPUT / "model_construction/logs" / f"{stage}_{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream):
        print("construction_job", stage, arguments, flush=True)
        return {
            "main": construct_main,
            "classifiers": construct_classifier,
            "priors": construct_priors,
        }[stage](*arguments)


def run(stage):
    validate_preparation()
    final = read_json(
        ROOT
        / "experiment_outputs/benchmark/collision/discovery/domain_comparison/diagonal_controls/finalization.json"
    )
    if (
        final["status"] != "complete_finalized_development_method"
        or final["covariance_mode"] != "diagonal"
    ):
        raise ValueError("Freeze the final method before constructing independent models")
    jobs, reports = available_jobs(stage), []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(build_job, stage, arguments) for arguments in jobs]
        for future in as_completed(futures):
            reports.append(future.result())
            write_json(
                OUTPUT / "model_construction" / f"required_{stage}.json",
                {
                    "status": "complete" if len(reports) == len(jobs) else "running",
                    "registered_scopes": len(jobs),
                    "completed_scopes": len(reports),
                    "workers": WORKERS,
                    "reports": reports,
                },
            )
            print("independent_models", stage, len(reports), "/", len(jobs), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("classifiers", "main", "priors"))
    run(parser.parse_args().stage)


if __name__ == "__main__":
    main()
