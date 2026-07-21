"""PEARL adaptation for held-out merge-like logical scenarios.

This package is intentionally independent from :mod:`sac_scenario_mining` so
that Stage 2 cannot silently alter the frozen Stage 1 environment.
"""

from .src.task_env import LogicalMergeEnv
from .src.task_spec import LogicalScenarioTaskSpec

__all__ = ["LogicalMergeEnv", "LogicalScenarioTaskSpec"]
