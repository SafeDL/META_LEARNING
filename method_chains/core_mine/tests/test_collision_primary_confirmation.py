from types import SimpleNamespace

import numpy as np

from method_chains.core_mine.collision_primary_confirmation import (
    SEEDS, _collision_cells, _paired,
)
from method_chains.core_mine.fvdm_source_robustness import SEEDS as DEVELOPMENT_SEEDS


def test_collision_cells_counts_distinct_mode_grid_cells():
    task = SimpleNamespace(features=np.asarray([
        [0.10, 0.10], [0.12, 0.12], [0.60, 0.60]]))
    rows = [{"index": "0", "mode": "cutin_braking", "ego_collision": "True"},
            {"index": "1", "mode": "cutin_braking", "ego_collision": "True"},
            {"index": "2", "mode": "fast_intrusion", "ego_collision": "True"}]
    assert _collision_cells(task, rows) == 2
    rows[-1]["ego_collision"] = "False"
    assert _collision_cells(task, rows) == 1


def test_new_seeds_and_paired_interval_contract():
    assert len(SEEDS) == 6
    assert not set(SEEDS) & set(DEVELOPMENT_SEEDS)
    result = _paired([1, 2, 3], np.random.default_rng(1))
    assert result["mean"] == 2
    assert result["bootstrap_95"][0] <= 2 <= result["bootstrap_95"][1]
