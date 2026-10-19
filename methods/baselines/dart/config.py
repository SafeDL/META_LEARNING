"""Predeclared method settings, shared by training and evaluation."""

from pathlib import Path

from methods.baselines.cg_brt.config import (
    BUDGET,
    INITIAL_QUERIES,
    JITTER,
    NOISE,
    PRIOR_DF,
    THRESHOLD,
)
from methods.baselines.hgt.config import SEEDS

ROOT = Path(__file__).resolve().parents[3] / "results/baselines/dart"
CANDIDATES_PER_RULE = 8
FANTASIES = 16
LOSS_FRACTION = 0.1
VALUE_TOLERANCE = 1e-8
WIDTH = 64
HEADS = 4
FEATURES = 16
CONTEXT_PER_FAMILY = 128
MAX_AMPLITUDE = 0.2
INITIAL_AMPLITUDE = 0.02
SCALE_NODES = 32
TRAINING_STEPS = 3200
VALIDATION_INTERVAL = 800
LEARNING_RATE = 3e-4
SUPPORT_COUNTS = (0, 5, 10, 20, 50, 100, 150)
QUERY_COUNT = 48
VALIDATION_BUDGET = 40
VALIDATION_INITIAL = 2
