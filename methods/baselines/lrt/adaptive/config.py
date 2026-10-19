"""Historical-only training and fixed adaptive evaluation settings."""

from ..config import ROOT as LRT_ROOT

ROOT = LRT_ROOT / "adaptive"
SEED = 11
STEPS = 1600
INTERVAL = 400
BATCH_SIZE = 16
THRESHOLDS = (0.80, 0.90, 0.95, 0.975, 0.99)
HORIZONS = (1, 5, 10, 20, 50, 100)
RETENTION = 0.9


def horizons(remaining):
    return sorted({b for b in (*HORIZONS, remaining) if 0 < b <= remaining})
