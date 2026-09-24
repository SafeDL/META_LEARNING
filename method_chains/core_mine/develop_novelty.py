"""Development-only check of a verified-event novelty penalty at B=50."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from method_chains.core_mine.config import CoreMineConfig
from method_chains.core_mine.experiment import run_campaign
from method_chains.core_mine import sparse_sut_experiment as sparse


ROOT = Path("results/method_chains/core_mine/studies/corrected_geometry")
PENALTIES = (0.0, 0.1, 0.25, 0.5, 0.75)


def main() -> None:
    sparse.configure_proposal("v5_corrected_geometry")
    frozen = json.loads((ROOT / "frozen_config.json").read_text(encoding="utf-8"))["config"]
    tasks = [task for seed in sparse.DEVELOPMENT_SEEDS
             for task in sparse.tasks_from_bank(sparse.load_or_build(seed), seed)]
    rows = []
    for penalty in PENALTIES:
        config = CoreMineConfig(frozen["residual_length"], frozen["residual_amplitude"],
                                frozen["observation_noise"], penalty)
        values = [row for task in tasks for row in run_campaign(task, "MeanResidual-Novelty", config)[0]
                  if row["budget"] == 50]
        rows.append({"penalty": penalty, "units": len(values),
                     "mean_CriticalCount50": float(np.mean([float(row["CriticalCount"]) for row in values])),
                     "mean_CollisionCount50": float(np.mean([float(row["CollisionCount"]) for row in values])),
                     "mean_CVS50": float(np.mean([float(row["CVS"]) for row in values])),
                     "mean_F50": float(np.mean([float(row["F"]) for row in values]))})
        print(rows[-1], flush=True)
    output = ROOT / "novelty_development.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
