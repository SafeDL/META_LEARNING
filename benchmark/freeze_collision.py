"""Validate all construction evidence before writing the one final protocol."""

from dataclasses import asdict
import json
import hashlib
from pathlib import Path

import torch

from controllers.highway.catalog import SUTS
from controllers.highway.learning import CHECKPOINT_ROOT
from controllers.highway.train import TRAINING, training_contract
from methods.baselines.fsbo.method import CONFIG as FSBO_CONFIG
from methods.baselines.pooled_ft.method import CONFIG as POOLED_CONFIG
from methods.common.response import archive_fingerprints
from methods.common.model_contract import validate_model_contract
from methods.gcrt.config import CONFIG

from .build_collision_models import model_path
from .collect import require_learning_freeze
from .construction import load_construction
from .protocol import DATA_ROOT, ROOT, freeze, method_contract
from .qualification import COURSES, QUALIFICATION_ROOT
from .scenarios import DOMAINS
from .studies import SCOPES


def main():
    require_learning_freeze()
    qualification, learned = [], []
    for sut in SUTS:
        report = json.loads((QUALIFICATION_ROOT / sut.sut_id / "qualification.json").read_text(encoding="utf-8"))
        if not report["qualified"] or len(report["courses"]) != 5 or not all(r["passed"] for r in report["courses"]):
            raise ValueError("Every registered object must pass the same five capability courses")
        qualification.append(report)
        if sut.family == "learning":
            training = json.loads((CHECKPOINT_ROOT / sut.training_run / "training.json").read_text(encoding="utf-8"))
            if (training["configuration"] != TRAINING or training["source_contract"] != training_contract()
                    or training["status"] != "qualified"):
                raise ValueError("Learning runs differ from the pre-target construction protocol")
            learned.append({"sut_id": sut.sut_id, "qualified": True,
                            "training_seed": sut.training_seed, "actual_timesteps": training["actual_timesteps"]})
    (QUALIFICATION_ROOT / "all.json").write_text(
        json.dumps({"status": "complete", "qualified": True, "objects": qualification}, indent=2), encoding="utf-8")
    (QUALIFICATION_ROOT / "learning.json").write_text(
        json.dumps({"status": "complete", "objects": learned}, indent=2), encoding="utf-8")
    models = []
    configurations = {"history_response": asdict(CONFIG), "fsbo": FSBO_CONFIG, "pooled_ft": POOLED_CONFIG}
    for kind in DOMAINS:
        all_history, all_development = load_construction(kind)
        for scope in SCOPES.values():
            history = [a for a in all_history if a.source_id in scope.history_ids]
            development = [a for a in all_development if a.source_id in scope.development_ids]
            methods = ("history_response",) if scope.name.startswith("swap_") else ("history_response", "fsbo", "pooled_ft")
            for method in methods:
                path = model_path(method, scope.name, kind)
                metadata = torch.load(path, weights_only=True, map_location="cpu")
                validate_model_contract(metadata, method, configuration=configurations[method])
                report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if (report["status"] != "complete" or metadata["history_fingerprints"] != archive_fingerprints(history)
                        or metadata["development_fingerprints"] != archive_fingerprints(development)
                        or metadata["held_family"] != scope.held_family
                        or report["curve"][-1]["step"] != configurations[method]["steps"]):
                    raise ValueError("A required scope model is incomplete or violates source isolation")
                models.append({"path": str(path.relative_to(ROOT)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    method = {"main": asdict(CONFIG), "fsbo": FSBO_CONFIG, "pooled_ft": POOLED_CONFIG,
              "models": models, "statistics": {"outer_unit": "implementation_or_training_lineage",
                                                  "bootstrap_repetitions": 2000, "seed": 1501}}
    evidence = {"interface": ROOT / "results/benchmark/collision/interface.json",
                "capability": QUALIFICATION_ROOT / "all.json", "policy_training": QUALIFICATION_ROOT / "learning.json",
                "development": ROOT / "results/benchmark/collision/development/report.json"}
    development = json.loads(evidence["development"].read_text(encoding="utf-8"))
    expected_configuration = json.loads(json.dumps({"main": asdict(CONFIG), "fsbo": FSBO_CONFIG,
                                                    "pooled_ft": POOLED_CONFIG}))
    core_models = {m["path"]: m["sha256"] for m in models if Path(m["path"]).parent.name == "all"}
    if (development["configuration"] != expected_configuration
            or development.get("source_contract") != method_contract()
            or development.get("model_hashes") != core_models):
        raise ValueError("Small-scale development must validate the current construction code and core weights")
    freeze(DATA_ROOT / "frozen_protocol.json", objects=SUTS, domains=DOMAINS, training=TRAINING,
           qualification={"courses": [asdict(course) for course in COURSES]}, method=method, evidence=evidence)


if __name__ == "__main__":
    main()
