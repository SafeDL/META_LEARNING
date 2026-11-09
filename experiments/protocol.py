"""Workspace paths and the physical execution horizon."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = ROOT / "artifacts"
DATA_ROOT = ARTIFACTS_ROOT / "data"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"


@dataclass(frozen=True)
class ExecutionConfig:
    episode_duration_s: float = 10.0


DEFAULT = ExecutionConfig()
