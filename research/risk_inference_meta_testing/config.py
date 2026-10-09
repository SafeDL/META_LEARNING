"""One fixed two-seed pilot on the released development cohort."""
from pathlib import Path

from methods.history_guided_testing.config import BUDGET


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COHORT = ROOT.parent / "behavior_response_testing" / "results" / "confirmation"
BACKBONES = ROOT.parent / "behavior_response_testing" / "results" / "models"
PREVIOUS = ROOT.parent / "risk_feedback_meta_testing"
SEEDS = (11, 23)
SUPPORT_COUNTS = (0, 10, 30, 50, 100, 150)
QUERY_COUNT = 256
TRAINING_STEPS = 1000
WARMUP_STEPS = 200
VALIDATION_INTERVAL = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MIN_STANDARD_DEVIATION = 0.0001
KERNEL_NUGGET = 0.000001
LOGIT_TO_PROBIT_SCALE = 1.6

