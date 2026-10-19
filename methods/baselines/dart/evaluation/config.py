"""One declared experiment; target results never select network checkpoints."""

from benchmark.config import SUT_IDS
from benchmark.data import SCENARIOS
from methods.baselines.dart.config import ROOT, SEEDS

INDEPENDENT_BANKS = SCENARIOS
INDEPENDENT_SEEDS = (860035, 860036)
VARIANTS = {
    "cg_brt": {},
    "full": {},
    "physical_kernel": {},
    "conventional_training": {},
    "greedy": {"lookahead": False},
    "unprotected": {"protection": False},
    "no_spatial_planning": {"spatial": False},
}
CONFIRMATION_VARIANTS = ("cg_brt", "full", "conventional_training", "greedy")
ACCEPTANCE_METRICS = ("H200", "high_risk_recall_pct")
RETENTION_FLOOR = 0.9


def bank_root(stage):
    return SCENARIOS


def pools(stage):
    return (0, 1) if stage == "development" else (2,)


def variants(stage):
    return tuple(VARIANTS) if stage == "development" else CONFIRMATION_VARIANTS
