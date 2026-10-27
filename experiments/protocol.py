"""Experiment roles, fixed budgets and strict family holdouts."""

import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_RECORDS_ROOT = ROOT / "test_records" / "collision"


@dataclass(frozen=True)
class Protocol:
    candidate_count: int = 512
    budget: int = 128
    initial_count: int = 8
    checkpoints: tuple[int, ...] = (16, 32, 64, 128)
    algorithm_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    environment_seeds: tuple[int, ...] = (81001,)
    physics_hz: int = 20
    decision_hz: int = 5
    episode_duration_s: float = 10.0

    def __post_init__(self):
        if not 0 < self.initial_count <= self.budget <= self.candidate_count:
            raise ValueError("Invalid candidate/init/query budget")
        if any(not self.initial_count <= b <= self.budget for b in self.checkpoints):
            raise ValueError("Checkpoints must be within the paid query budget")
        if self.physics_hz % self.decision_hz or self.episode_duration_s <= 0:
            raise ValueError("Invalid execution frequency")


DEFAULT = Protocol()


def runtime_environment():
    import torch

    packages = (
        "numpy",
        "scipy",
        "torch",
        "gymnasium",
        "stable-baselines3",
        "highway-env",
        "scikit-learn",
        "matplotlib",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "packages": {name: version(name) for name in packages},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def check_roles(history, development, *, held_family=None):
    if not history or not development:
        raise ValueError("Model construction requires historical and development objects")
    if any(s.role != "history" for s in history):
        raise ValueError("Only historical objects may construct representation and priors")
    if any(s.role != "development" for s in development):
        raise ValueError("Final targets cannot enter development or checkpoint selection")
    if held_family and any(s.family == held_family for s in (*history, *development)):
        raise ValueError("Held family must be excluded from all construction/development")
