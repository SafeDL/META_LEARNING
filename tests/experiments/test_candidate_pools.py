from itertools import product

import numpy as np
import pytest

from experiments.joint_validation import Config, target_bank
from experiments.scenarios import (
    DOMAINS,
    TEST_SCENARIOS,
    Domain,
    candidate_count,
    candidate_pool,
    difficult,
    features,
    validate,
)


@pytest.mark.parametrize(
    "kind,count,dimension",
    [
        ("car_following", 1000, 3),
        ("front_brake", 2000, 6),
        ("cut_out", 2000, 6),
        ("lane_change_rear", 2000, 6),
    ],
)
def test_candidate_pools_keep_natural_dimensions_and_all_feasible_difficulties(
    kind, count, dimension
):
    scenes = candidate_pool(kind)
    assert candidate_count(kind) == len(scenes) == count
    assert len({s.scenario_id for s in scenes}) == count
    assert len({tuple(s.parameters.items()) for s in scenes}) == count
    assert scenes == candidate_pool(kind)
    assert scenes != candidate_pool(kind, replicate=20)
    assert features(scenes).shape == (count, dimension)
    assert {difficult(s) for s in scenes} == {False, True}
    for scene in scenes:
        validate(scene)


def test_following_grid_has_ten_strata_per_dimension_and_all_combinations():
    coordinates = features(candidate_pool("car_following"))
    axes = [np.unique(column) for column in coordinates.T]
    for axis in axes:
        assert len(axis) == 10
        np.testing.assert_array_equal(np.floor(axis * 10), np.arange(10))
    assert set(map(tuple, coordinates)) == set(product(*axes))


def test_two_dimensional_grid_has_one_hundred_candidates(monkeypatch):
    domain = Domain(
        {"ego_speed_mps": (18, 34), "gap_m": (8, 65)}, {"ego_speed_mps": 1, "gap_m": 2}, {}
    )
    monkeypatch.setitem(DOMAINS, "car_following", domain)
    scenes = candidate_pool("car_following")
    assert len(scenes) == candidate_count("car_following") == 100
    assert features(scenes).shape == (100, 2)


def test_constrained_sobol_pools_are_nested_for_scale_comparisons():
    small = candidate_pool("cut_out", 1000, replicate=30)
    large = candidate_pool("cut_out", 4000, replicate=30)
    assert small == large[:1000]


def test_current_protocol_routes_each_scene_to_its_own_candidate_count(monkeypatch, tmp_path):
    from experiments import joint_validation

    config = Config()
    assert (config.initial_count, config.budget, config.algorithm_seeds) == (20, 200, (11,))
    assert config.scenarios == TEST_SCENARIOS

    def collect(sut_id, candidates, root, *, environment_seed, current_registration):
        assert environment_seed == 81001
        assert current_registration
        return candidates

    monkeypatch.setattr(joint_validation, "collect_bank", collect)
    from controllers.highway.catalog import INDEPENDENT_SUTS

    for kind in config.scenarios:
        assert len(target_bank(tmp_path, INDEPENDENT_SUTS[0], kind, config)) == candidate_count(
            kind
        )
    with pytest.raises(ValueError, match="current test scenarios"):
        Config(scenarios=("cut_in",))
    with pytest.raises(ValueError, match="current test scenario"):
        candidate_pool("cut_in_interaction")
