"""Fixed driving benchmark settings and result locations."""

from pathlib import Path

from methods.baselines.hgt.config import (
    BOUNDS,
    BUDGET,
    MEASUREMENT,
    PROFILES,
    SEEDS,
    TARGET,
    TEMPLATES,
    build_spec,
)
from controllers.highway.registry import build_spec_factory


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "benchmark"
SUT_IDS = (
    "idm_ref",
    "fvdm_target",
    "mobil_ref_v2",
    "vi_ttc_ref_audit_v4",
    "mcts_cv_ref_audit_v4",
    "ppo_ref_v2",
)
POOL_SEEDS = ((860031, 860032), (860033, 860034))
WORKERS = 6
AUDIT_INDICES = (0, 1, 2, 3, 1024, 1025, 1026, 1027)
FEEDBACK = "queried continuous risk and same-execution ego collision"


def sut_spec(sut_id):
    return build_spec(TARGET) if sut_id == "fvdm_target" else build_spec_factory(sut_id)
