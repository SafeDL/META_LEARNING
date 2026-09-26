from highway_sim_env.envs.cutin_env import CutInScenario
from highway_sim_env.envs.external_cutin import ExternalCutInEnv
from replications.highway_sut_selection.runner import common_scenarios


def _speed_after(action: int) -> float:
    env = ExternalCutInEnv(CutInScenario(35, 0, "slow_lead_following"))
    try:
        env.reset(seed=5)
        env.step(action)
        return env.vehicle.target_speed
    finally:
        env.close()


def test_external_longitudinal_actions_change_target_speed():
    assert _speed_after(3) > _speed_after(4)


def test_external_lateral_action_changes_target_lane():
    env = ExternalCutInEnv(CutInScenario(35, 0, "slow_lead_following"))
    try:
        env.reset(seed=5)
        env.step(2)
        assert env.vehicle.target_lane_index[2] == 1
    finally:
        env.close()


def test_common_protocol_has_six_balanced_functional_modes():
    scenarios = common_scenarios(96, 7)
    assert len(scenarios) == 96
    assert {scenario.mode for scenario in scenarios} == {
        "fast_intrusion",
        "cutin_braking",
        "lead_braking",
        "stop_and_go",
        "slow_lead_following",
        "passing_cutin",
    }
    observed_modes = {scenario.mode for scenario in scenarios}
    assert all(
        sum(scenario.mode == mode for scenario in scenarios) == 16
        for mode in observed_modes
    )


def test_adjacent_side_by_side_vehicles_are_not_a_near_miss():
    env = ExternalCutInEnv(CutInScenario(0, 0, "fast_intrusion"))
    try:
        env.reset(seed=5)
        env._update_safety_metrics()
        assert env.external_result().min_distance >= 2.0 - 1e-6
        assert not env.external_result().near_miss
    finally:
        env.close()


def test_same_lane_sub_meter_clearance_is_a_near_miss():
    env = ExternalCutInEnv(CutInScenario(10, 0, "lead_braking"))
    try:
        env.reset(seed=5)
        env.scheduled_vehicle.position = env.vehicle.position + (5.8, 0.0)
        env._update_safety_metrics()
        assert 0.0 < env.external_result().min_distance < 1.0
        assert env.external_result().near_miss
    finally:
        env.close()
