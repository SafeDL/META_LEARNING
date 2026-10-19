"""Measured scenario data; execution times are read only by evaluation code."""

from pathlib import Path

import numpy as np

from methods.baselines.hgt.io import read_rows

PROJECT = Path(__file__).resolve().parents[1]
HISTORY = PROJECT / "data/history"
SCENARIOS = PROJECT / "data/scenarios"


def bank_path(pool, sut):
    return SCENARIOS / f"pool_{pool}" / sut / "responses.npz"


def measured_seconds(pool, sut, scenario_ids):
    rows = read_rows(bank_path(pool, sut).with_name("measurements.jsonl"))
    mapping = {row["scenario_id"]: float(row["elapsed_s"]) for row in rows}
    values = np.asarray([mapping[str(i)] for i in scenario_ids])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Archived execution timings must be finite and positive")
    return values
