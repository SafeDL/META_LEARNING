"""Frozen baseline settings and relocated inference inputs."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASELINE_INPUTS = ROOT / "results/baselines/response_baselines"
HISTORY_MODELS = BASELINE_INPUTS / "history_models"
BEHAVIOR_MODELS = BASELINE_INPUTS / "behavior_models"
SOURCE_SNAPSHOT = BASELINE_INPUTS / "source_inputs.zip"
BUDGET = 200
SOURCE_WEIGHTS = (1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4)
RESPONSE_LENGTH = 0.05
PHYSICAL_LENGTH = 0.2
PHYSICAL_WEIGHT = 0.1
GP_AMPLITUDE = 0.25
REGULARIZATION_VARIANCE = 0.05
