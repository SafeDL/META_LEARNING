"""Locked settings; the event is measured risk strictly above 0.5."""
from methods.baselines.hgt.config import BUDGET, FEEDBACK

THRESHOLD = .5
INITIAL_QUERIES = 10
NOISE = FEEDBACK["noise_variance"]
JITTER = 1e-8
PRIOR_DF = 5
SET_SAMPLES = 128
SCALE_NODES = 32
CDF_NODES = 64
