"""One fixed five-seed benchmark, with no method-search stages."""

from pathlib import Path

from methods.baselines.cg_brt.config import (
    BUDGET,
    INITIAL_QUERIES,
)
from methods.baselines.hgt.config import SEEDS

PROJECT = Path(__file__).resolve().parents[4]
ROOT = PROJECT / "results/baselines/cg_brt"
POLICY = "cg_brt"
METRICS = ("H200", "high_risk_recall_pct")
