"""Prospective axes must affect the physical scene, not just its manifest."""

from copy import deepcopy
from itertools import combinations

import pytest

from highway_sim_env.envs.fbrt_unified_env import FBRTUnifiedEnv
from methods.failure_memory_regression.pattern_memory import active_values
from methods.failure_memory_regression.prepare_scenario_sampling import compile_manifest
from sut_algorithms.highway_env.registry import build_spec_factory


def _initial_physics(case):
    env = FBRTUnifiedEnv(build_spec_factory("nl_v0"), case)
    try:
        env.reset(seed=case["simulator_seed"])
        return tuple((role, tuple(float(x) for x in actor.position),
                      float(actor.speed),
                      *(getattr(actor, name, None) for name in (
                          "event_start_s", "lane_change_duration_s",
                          "deceleration_mps2", "hold_s",
                          "restart_acceleration_mps2", "brake_after_measured_merge_s")))
                     for role, actor in sorted(env.actors.items()))
    finally:
        env.close()


@pytest.fixture(scope="module")
def scenario_rows():
    return compile_manifest()


def test_sampling_counts_and_pairwise_coverage(scenario_rows):
    rows = scenario_rows
    expected = {"S01": 2048, "S02": 2048, "S03": 2048, "S04": 4096,
                "S05": 2048, "S06": 1024, "S08": 4096, "S09": 2048}
    assert len(rows) == 19456
    assert len({row["scenario_id"] for row in rows}) == len(rows)
    families = {row["catalogue_id"] for row in rows}
    assert families == set(expected)
    for family in families:
        local = [row for row in rows if row["catalogue_id"] == family]
        assert len(local) == expected[family]
        assert all(set(row["parameter_groups"]) == set(row["active_parameters"])
                   for row in local)
        assert all(not (set(row["active_parameters"]) & set(row["fixed_context"]))
                   for row in local)
        points = [active_values(row) for row in local]
        assert all(len(point) == len(local[0]["research_bounds"]) for point in points)
        assert all(all(0 <= value <= 1 for value in point) for point in points)
        for x, y in combinations(range(len(points[0])), 2):
            cells = {(min(int(point[x] * 16), 15), min(int(point[y] * 16), 15))
                     for point in points}
            assert len(cells) == 256, (family, x, y)


@pytest.mark.parametrize("family", ["S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09"])
def test_each_v3_axis_changes_initial_physics(family, scenario_rows):
    case = next(row for row in scenario_rows if row["catalogue_id"] == family)
    baseline = _initial_physics(case)
    for name, (lower, upper) in case["research_bounds"].items():
        changed = deepcopy(case)
        original = changed["active_parameters"][name]
        delta = (upper - lower) * 0.1
        changed["active_parameters"][name] = original + delta if original + delta < upper else original - delta
        assert _initial_physics(changed) != baseline, (family, name)
