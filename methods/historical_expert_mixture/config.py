"""Locked settings for evidence-conditioned historical response mixtures."""
from methods.history_guided_testing.config import BUDGET, FEEDBACK


INITIAL_QUERIES = 10
NEIGHBORS = 8
NEIGHBOR_LENGTH = .15
NOISE = FEEDBACK["noise_variance"]
JITTER = 1e-8

