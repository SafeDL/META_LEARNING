"""Count which v4 validation near-miss labels have TTC support at B=50."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/method_chains/core_mine/studies/sparse_continuous_suts")
VALIDATION_SEEDS = (20280317, 20280331, 20280414)


def _counts(collision: np.ndarray, near_miss: np.ndarray, ttc: np.ndarray) -> dict[str, int]:
    return {
        "ego_collisions": int(collision.sum()),
        "archived_near_misses": int(near_miss.sum()),
        "near_misses_with_ttc_below_1_5s": int((near_miss & (ttc < 1.5)).sum()),
        "near_misses_distance_only": int((near_miss & ~(ttc < 1.5)).sum()),
    }


def main() -> None:
    banks = {}
    for seed in VALIDATION_SEEDS:
        path = ROOT / "banks" / "sparse_continuous" / f"sparse_sut_bank_{seed}.npz"
        with np.load(path, allow_pickle=False) as bank:
            banks[seed] = {key: bank[key].copy() for key in ("sut_names", "ego_collision", "near_miss", "min_ttc")}
    all_counts = _counts(
        np.concatenate([banks[s]["ego_collision"].ravel() for s in VALIDATION_SEEDS]),
        np.concatenate([banks[s]["near_miss"].ravel() for s in VALIDATION_SEEDS]),
        np.concatenate([banks[s]["min_ttc"].ravel() for s in VALIDATION_SEEDS]),
    )
    selected: dict[str, list[dict[str, int]]] = defaultdict(list)
    with (ROOT / "validate" / "records.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["budget"]) != 50 or int(row["repeat"]) != 0:
                continue
            bank = banks[int(row["seed"])]
            target = list(bank["sut_names"].astype(str)).index(row["heterogeneity"])
            indices = np.asarray([int(value) for value in row["queried_indices"].split(";")])
            selected[row["method"]].append(_counts(
                bank["ego_collision"][target, indices],
                bank["near_miss"][target, indices],
                bank["min_ttc"][target, indices],
            ))
    report = {
        "scope": "v4 validation; 3 seeds x 3 target SUTs x 500 candidates",
        "threshold_seconds": 1.5,
        "all_candidates": all_counts,
        "selected_B50": {
            method: {key: sum(unit[key] for unit in units) for key in units[0]}
            for method, units in selected.items()
        },
        "interpretation": "Distance-only labels are not verified hazards; the archived distance proxy ignores lateral vehicle width.",
    }
    path = ROOT / "label_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
