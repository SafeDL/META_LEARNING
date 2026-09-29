"""Prepare an eight-family, one-context development bank without simulation."""

from __future__ import annotations

import json
from pathlib import Path

from methods.failure_memory_regression import bidirectional as fbrt
from methods.failure_memory_regression.prepare_role_gated_generalization_confirmation import (
    BASELINES, CHAINS, FAMILIES, METHODS, PRIMARY,
)


ROOT = Path("results/method_chains/failure_memory_regression/single_context_grid_development")
SPLIT = "single_context_grid_development"
RESOLUTION = 11
SEED = 4179940


def main() -> None:
    summary = {}
    for chain, builds in CHAINS.items():
        rows = fbrt.freeze(
            ROOT / chain,
            split=SPLIT,
            resolution=RESOLUTION,
            seed=SEED,
            builds=builds,
            methods=METHODS,
            families=FAMILIES,
            primary_method=PRIMARY,
            primary_baselines=BASELINES,
        )
        assert len(rows) == len(FAMILIES) * RESOLUTION**2
        assert len({row["context_id"] for row in rows}) == len(FAMILIES)
        summary[chain] = {
            "scenarios": len(rows),
            "contexts_per_family": 1,
            "planned_physical_episodes": len(rows) * len(builds),
        }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
