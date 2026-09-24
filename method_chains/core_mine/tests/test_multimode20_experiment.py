import numpy as np

from method_chains.core_mine.acquisition import marginal_scores
from method_chains.core_mine.multimode20_experiment import (
    BOUNDS, SEEDS, _cell, proposal,
)
from method_chains.core_mine.single_lane_opportunity_pilot import SEED as PILOT_SEED


def test_frozen_multimode_pool_and_seed_disjointness():
    assert PILOT_SEED not in SEEDS
    anchors, modes, controls, _ = proposal(SEEDS[0])
    assert anchors.shape == (320, 2)
    assert controls.shape == (320, 2)
    assert {mode: int(np.sum(modes == mode)) for mode in BOUNDS} == {
        mode: 64 for mode in BOUNDS}


def test_collision_cell_fixed_physical_bounds_and_overflow():
    assert _cell("lead_braking", 8.0, -7.0) == ("lead_braking", 0, 0)
    assert _cell("lead_braking", 30.0, 8.0) == ("lead_braking", 3, 3)
    assert _cell("lead_braking", 7.0, 9.0) == ("lead_braking", -1, 4)


def test_marginal_references_only_eligible_source_safe_suite():
    features = np.asarray([[0.0], [0.2], [0.0]])
    modes = np.asarray(["a", "a", "b"])
    risk = np.ones(3)
    scores = marginal_scores(features, modes, [], [], risk, risk, 0.0,
                             evaluation_indices=np.asarray([0, 1]))
    assert scores.shape == (3,)
    assert scores[0] > 0
    assert scores[2] == 0
