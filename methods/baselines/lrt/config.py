"""Fixed research protocol; target results do not select these settings."""

from pathlib import Path

from methods.baselines.dart.config import SEEDS
from methods.common.config import JITTER, NOISE

ROOT = Path(__file__).resolve().parents[3] / "results/baselines/lrt"
MODELS = ((0.5, 0.10), (1.0, 0.25), (2.0, 0.50))
NOMINAL = 1
BUDGET = 200
INITIAL = 10
TRAIN_BUDGET = 160
TRAIN_INITIAL = 8
TRAIN_STATES = (8, 16, 40, 80, 120, 152)
VALID_BUDGET = 40
VALID_INITIAL = 2
VALID_STATES = (2, 4, 10, 20, 30, 38)
ARCHIVES = ("complete", "sparse", "cutin_gap", "brake_gap")
TRAIN_MODES = ("full", "global_sources", "no_archive_covariance", "no_target_component")
VARIANTS = (*TRAIN_MODES, "no_value_learning", "nominal_guard", "unprotected")
TRAIN_STEPS = 3200
REFINE_STEPS = 800
INTERVAL = 800
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
RETENTION = 0.9
FANTASIES = 16
LOSS_FRACTION = 0.1
VALUE_TOLERANCE = 1e-8


def options(mode):
    return {
        "local": mode != "global_sources",
        "archive_uncertainty": mode != "no_archive_covariance",
        "target_component": mode != "no_target_component",
    }
