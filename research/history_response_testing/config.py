from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
ARCHIVES = ROOT / "archives"
CONFIRMATION = OUTPUT / "final_confirmation"
BUDGET = 200
SEEDS = (11, 23, 37, 53, 71)
SOURCE_WEIGHTS = (1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4)
RESPONSE_LENGTH = 0.05
PHYSICAL_LENGTH = 0.2
PHYSICAL_WEIGHT = 0.1
GP_AMPLITUDE = 0.25
REGULARIZATION_VARIANCE = 0.05
METHODS = (
    "candidate",
    "main",
    "ras_frt_uq",
    "class_rank",
    "risk_only",
    "calibrated_only",
)
