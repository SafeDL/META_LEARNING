"""Workspace paths and the physical execution horizon."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"


@dataclass(frozen=True)
class ExecutionConfig:
    episode_duration_s: float = 10.0


DEFAULT = ExecutionConfig()


def episode_duration(kind):
    return (
        14.0
        if kind == "car_following_braking_onset"
        else DEFAULT.episode_duration_s
    )
