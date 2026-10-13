"""One fixed five-seed benchmark, with no method-search stages."""
from pathlib import Path

from methods.concordance_gated_bayesian_risk_transfer.config import BUDGET, INITIAL_QUERIES
from methods.history_guided_testing.config import SEEDS

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results/concordance_gated_bayesian_risk_transfer"
POLICY = "cg_brt"
METRICS = ("high_risk_area_pct", "high_risk_recall_pct", "H50", "H100", "H200")
