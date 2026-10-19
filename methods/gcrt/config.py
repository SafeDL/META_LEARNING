"""Fixed single-seed construction and controller-held-out evaluation protocol."""

from pathlib import Path

ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "results/gcrt/full_horizon"
ROOT = Path(__file__).resolve().parents[2] / "results/gcrt"
BASELINE_ROOT = Path(__file__).resolve().parents[2] / "results/baselines/lrt/by_family"
SEED = 11
BUDGET = 200
INITIAL = 10
FANTASIES = 16
TOLERANCE = 1e-8
HORIZONS = (1, 5, 10, 20, 50, 100)
SEGMENT = 5
TRAIN_STEPS = 800
UPDATE_STEPS = 400
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
ALPHA = 0.05
REUSE = (1, 10, 50, 100, 200)
ROLES = {
    "training": ("vi_ttc_ref_audit_v4", "fvdm_target"),
    "calibration": ("mcts_cv_ref_audit_v4", "ppo_ref_v2"),
    "evaluation": ("mobil_ref_v2", "idm_ref"),
}


def horizons(remaining):
    return sorted({b for b in (*HORIZONS, remaining) if 0 < b <= remaining})
