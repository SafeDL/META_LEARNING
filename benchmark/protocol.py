"""Experiment roles, fixed budgets and strict family holdouts."""

from dataclasses import asdict, dataclass
import json
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "collision"
RESULT_ROOT = ROOT / "results" / "gcrt" / "collision"


@dataclass(frozen=True)
class Protocol:
    candidate_count: int = 512
    budget: int = 128
    initial_count: int = 8
    horizon: int = 5
    checkpoints: tuple[int, ...] = (16, 32, 64, 128)
    algorithm_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    environment_seeds: tuple[int, ...] = (81001,)
    physics_hz: int = 20
    decision_hz: int = 5
    episode_duration_s: float = 10.0
    conditions: tuple[str, ...] = ("reuse", "new", "hard")

    def __post_init__(self):
        if not 0 < self.initial_count <= self.budget <= self.candidate_count:
            raise ValueError("Invalid candidate/init/query budget")
        if any(not self.initial_count <= b <= self.budget for b in self.checkpoints):
            raise ValueError("Checkpoints must be within the paid query budget")
        if self.horizon < 1 or self.physics_hz % self.decision_hz or self.episode_duration_s <= 0:
            raise ValueError("Invalid lookahead or execution frequency")


DEFAULT = Protocol()


def runtime_environment():
    import torch
    packages = ("numpy", "scipy", "torch", "gymnasium", "stable-baselines3",
                "highway-env", "scikit-learn", "matplotlib")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(), "packages": {name: version(name) for name in packages},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def method_contract():
    paths = ("methods/gcrt/posterior.py", "methods/gcrt/prior.py", "methods/gcrt/selection.py",
             "methods/gcrt/train.py",
             "methods/common/response.py", "methods/common/classification.py", "methods/common/gp_classifier.py",
             "methods/common/model_contract.py",
             "methods/baselines/fsbo/method.py", "methods/baselines/pooled_ft/method.py",
             "benchmark/schema.py", "benchmark/scenarios.py", "benchmark/query.py",
             "benchmark/storage.py", "benchmark/construction.py", "benchmark/executions.py",
             "benchmark/build_collision_models.py", "benchmark/protocol.py",
             "methods/baselines/response_baselines/collision.py", "benchmark/collision_runs.py",
             "benchmark/collision_metrics.py", "benchmark/studies.py", "benchmark/execution_events.py",
             "benchmark/certify_collisions.py", "benchmark/report_collision.py",
             "benchmark/collect.py", "benchmark/evaluate_collision.py", "benchmark/freeze_collision.py",
             "benchmark/develop_collision.py",
             "benchmark/run_collision.py",
             "environment.yml")
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def allowed_objects(objects, held_family=None):
    return tuple(s for s in objects if s.family != held_family)


def check_roles(history, development, *, held_family=None):
    if not history or not development:
        raise ValueError("Model construction requires historical and development objects")
    if any(s.role != "history" for s in history):
        raise ValueError("Only historical objects may construct representation and priors")
    if any(s.role != "development" for s in development):
        raise ValueError("Final targets cannot enter development or checkpoint selection")
    if held_family and any(s.family == held_family for s in (*history, *development)):
        raise ValueError("Held family must be excluded from all construction/development")


def freeze(path, *, objects, domains, training, qualification, method, evidence):
    """Write a new immutable protocol after the required qualification/dev gates."""
    required = {"interface", "capability", "policy_training", "development"}
    if set(evidence) != required or not all(Path(p).is_file() for p in evidence.values()):
        raise ValueError("Freeze needs existing evidence for every pre-evaluation gate")
    reports = {key: json.loads(Path(value).read_text(encoding="utf-8"))
               for key, value in evidence.items()}
    if reports["interface"].get("status") != "passed":
        raise ValueError("Interface acceptance has not passed")
    if not reports["capability"].get("qualified"):
        raise ValueError("Capabilities have not qualified")
    learned = {s.sut_id for s in objects if s.family == "learning"}
    runs = reports["policy_training"].get("objects", [])
    if {s["sut_id"] for s in runs} != learned or not all(s.get("qualified") for s in runs):
        raise ValueError("All five independent learning runs must qualify before final evaluation")
    if reports["development"].get("status") != "complete":
        raise ValueError("Small-scale development validation has not completed")
    from .studies import CONTINUOUS_TASKS, SCOPES, tasks
    from .executions import execution_contract
    manifest = {"protocol": asdict(DEFAULT), "objects": [asdict(s) for s in objects],
                "runtime_environment": runtime_environment(),
                "sut_contracts": {s.sut_id: execution_contract(s) for s in objects},
                "domains": {k: asdict(v) for k, v in domains.items()},
                "training": training, "qualification": qualification,
                "method": method, "evidence": {k: str(v) for k, v in evidence.items()},
                "source_contract": method_contract(),
                "model_scopes": {name: asdict(scope) for name, scope in SCOPES.items()},
                "study_tasks": [asdict(task) for task in tasks()],
                "continuous_tasks": CONTINUOUS_TASKS}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def validate_sut_freeze(manifest):
    """Controller code and selected policy weights are part of the final experiment."""
    from controllers.highway.catalog import SUTS
    from .executions import execution_contract
    if manifest["objects"] != json.loads(json.dumps([asdict(s) for s in SUTS])):
        raise ValueError("Registered systems changed after final protocol freezing")
    current = {s.sut_id: execution_contract(s) for s in SUTS}
    if manifest["sut_contracts"] != json.loads(json.dumps(current)):
        raise ValueError("Controller implementation or selected policy weights changed after freezing")
