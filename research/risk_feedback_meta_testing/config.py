"""Fixed development settings, separated from the frozen candidates."""
from pathlib import Path

from methods.history_guided_testing.config import BUDGET, TEMPLATES
from research.behavior_response_testing.train import SEEDS


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COHORT = ROOT.parent / "behavior_response_testing" / "results" / "confirmation"
MODELS = ROOT.parent / "behavior_response_testing" / "results" / "models"
CURRENT_CONFIRMATION = (
    ROOT.parent / "risk_conditioned_response_testing" / "results" / "confirmation"
)
SPLIT_RANGES = {"training": range(12), "validation": range(12, 18),
                "development_holdout": range(18, 24)}
SUPPORT_COUNTS = (0, 10, 30, 50, 100, 150)
QUERY_COUNT = 256
TRAINING_STEPS = 4000
VALIDATION_INTERVAL = 250
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EXPLOITATION_ROOTS = 8
UNCERTAINTY_ROOTS = 8
INTERIOR_QUADRATURE_POINTS = 16
METHODS = (
    "supervised_greedy", "meta_greedy", "supervised_lookahead",
    "meta_lookahead", "meta_unconstrained",
)


def split_names():
    return {
        role: [f"{controller}_{index:02d}"
               for controller in ("idm", "fvdm") for index in indices]
        for role, indices in SPLIT_RANGES.items()
    }
