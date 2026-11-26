"""Versioned execution identity, independent of archived data locations."""

import hashlib
from importlib.metadata import version
from pathlib import Path

from simulator.highway.timing import timing_contract

IMPLEMENTATION_VERSION = "ads-native-highway-v2"
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def runtime_contract(sut, *, checkpoint_path=None):
    paths = [
        "simulator/highway/collision_env.py",
        "simulator/highway/timing.py",
        "simulator/highway/observation.py",
        "controllers/highway/control.py",
        "controllers/highway/decision_interface.py",
        "controllers/highway/factory.py",
        "controllers/highway/contracts.py",
        "experiments/scenarios.py",
    ]
    module = {
        "rule": "rule",
        "optimization": "optimization",
        "search": "search",
        "learning": "learning",
    }[sut.family]
    paths.append(f"controllers/highway/{module}.py")
    if sut.family in ("optimization", "search"):
        paths.append("controllers/highway/planning.py")
    result = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "sut_id": sut.sut_id,
        "parameters": sut.parameters,
        "highway_env": version("highway-env"),
        "timing": timing_contract(),
        "source_files": {
            p: hashlib.sha256((SOURCE_ROOT / p).read_bytes()).hexdigest() for p in paths
        },
    }
    if sut.family == "learning":
        from .learning import CHECKPOINT_ROOT

        checkpoint_path = (
            CHECKPOINT_ROOT / sut.training_run / "best.zip"
            if checkpoint_path is None
            else Path(checkpoint_path)
        )
        result["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        result["stable_baselines3"] = version("stable-baselines3")
    return result
