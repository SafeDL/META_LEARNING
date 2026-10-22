"""Reproduce saved model validation without SUT calls or checkpoint writes."""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from benchmark.build_collision_models import model_path
from benchmark.construction import load_construction
from benchmark.scenarios import DOMAINS
from benchmark.studies import SCOPES
from methods.baselines.fsbo import method as fsbo
from methods.baselines.pooled_ft import method as pooled
from methods.common.model_contract import validate_model_contract
from methods.common.response import archive_fingerprints
from methods.gcrt.config import CONFIG
from methods.gcrt.train import load_encoder, validate


def verify(method, scope, kind, history, development):
    path = model_path(method, scope.name, kind)
    report_path = path.with_suffix(".json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loader = {"history_response": load_encoder, "fsbo": fsbo.load, "pooled_ft": pooled.load}[method]
    model, metadata = loader(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_config = {"history_response": asdict(CONFIG), "fsbo": fsbo.CONFIG, "pooled_ft": pooled.CONFIG}[method]
    validate_model_contract(metadata, method, configuration=expected_config)
    assert report["status"] == "complete" and report["held_family"] == metadata["held_family"] == scope.held_family
    assert report["config"] == json.loads(json.dumps(expected_config))
    assert set(metadata["history_ids"]) == {a.source_id for a in history} and len(metadata["history_ids"]) == len(history)
    assert set(metadata["development_ids"]) == {a.source_id for a in development} and len(metadata["development_ids"]) == len(development)
    assert metadata["history_fingerprints"] == archive_fingerprints(history)
    assert metadata["development_fingerprints"] == archive_fingerprints(development)
    steps, interval = expected_config["steps"], expected_config["validation_interval"]
    assert [row["step"] for row in report["curve"]] == sorted(set([*range(interval, steps + 1, interval), steps]))
    if method == "history_response":
        best = min(report["curve"], key=lambda row: (row["validation"]["collision_nll"], row["validation"]["brier"]))
        assert metadata["step"] == best["step"]
        recorded = best["validation"]["collision_nll"]
        reproduced = validate(model, history, development, CONFIG)["collision_nll"]
        assert all(math.isfinite(row["validation"]["collision_nll"]) for row in report["curve"])
    else:
        best = min(report["curve"], key=lambda row: row["development_nll"])
        recorded = best["development_nll"]
        scores = []
        for target in development:
            indices = np.random.default_rng(expected_config["seed"] + 1).choice(
                len(target.scenarios), expected_config["support_count"] + expected_config["query_count"], replace=False)
            support, query = indices[:expected_config["support_count"]], indices[expected_config["support_count"]:]
            adapted = copy.deepcopy(model)
            tune = fsbo.fit_target if method == "fsbo" else pooled.fine_tune
            tune(adapted, target.x[support], target.collision[support],
                 steps=expected_config["adaptation_steps"], learning_rate=expected_config["adaptation_rate"])
            with torch.no_grad():
                if method == "fsbo":
                    probability = fsbo.collision_probabilities(adapted, target.x[support], target.collision[support], target.x[query])
                    score = torch.nn.functional.binary_cross_entropy(probability, torch.as_tensor(target.collision[query], dtype=probability.dtype))
                else:
                    score = torch.nn.functional.binary_cross_entropy_with_logits(adapted(target.x[query]), torch.as_tensor(target.collision[query], dtype=torch.float32))
                scores.append(float(score))
        reproduced = float(np.mean(scores))
        assert all(math.isfinite(row["development_nll"]) for row in report["curve"])
    assert math.isfinite(reproduced) and math.isclose(reproduced, recorded, rel_tol=1e-5, abs_tol=1e-7), (method, scope.name, kind, reproduced, recorded)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return {"method": method, "scope": scope.name, "scenario": kind, "checkpoint": str(path),
            "checkpoint_sha256": digest, "training_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "completed_steps": steps, "best_curve_step": best["step"], "history_ids": metadata["history_ids"],
            "development_ids": metadata["development_ids"], "history_fingerprints_verified": True,
            "development_fingerprints_verified": True, "model_contract_verified": True,
            "recorded_validation_nll": recorded, "reproduced_validation_nll": reproduced}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("core", "complete"))
    arguments = parser.parse_args()
    started = time.perf_counter()
    selected = [SCOPES["all"]] if arguments.stage == "core" else list(SCOPES.values())
    rows, errors, pending = [], [], []
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for kind in DOMAINS:
            historical, all_development = load_construction(kind)
            for scope in selected:
                history = [a for a in historical if a.source_id in scope.history_ids]
                development = [a for a in all_development if a.source_id in scope.development_ids]
                methods = ["history_response"] if scope.name.startswith("swap_") else ["history_response", "fsbo", "pooled_ft"]
                for method in methods:
                    path = model_path(method, scope.name, kind)
                    if not path.exists() or not path.with_suffix(".json").exists():
                        pending.append(str(path))
                        continue
                    try:
                        rows.append(verify(method, scope, kind, history, development))
                        print("verified", method, scope.name, kind, flush=True)
                    except Exception as error:
                        errors.append({"path": str(path), "error": repr(error)})
    expected = 12 if arguments.stage == "core" else 156
    complete = not errors and not pending and len(rows) == expected
    result = {"status": "complete" if complete else "failed" if errors else "partial",
              "scope": f"{arguments.stage} model construction and validation replay only; final study completion is separate.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "expected_models": expected,
              "verified_models": len(rows), "additional_physical_calls": 0, "verification_wall_s": time.perf_counter() - started,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "nll_tolerance": {"relative": 1e-5, "absolute": 1e-7}, "models": rows, "pending": pending, "errors": errors}
    output = Path(__file__).with_name("core_model_verification.json" if arguments.stage == "core" else "full_model_verification.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"], len(rows), "/", expected, "models;", len(errors), "errors", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
