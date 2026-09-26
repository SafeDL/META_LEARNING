from highway_sim_env.envs.cutin_env import CutInScenario
from highway_sim_env.envs.single_lane_longitudinal import (
    SingleLaneExternalCutInEnv, SingleLaneLongitudinalEnv,
)
from methods.core_mine.fvdm_revision_pilot import BUILDS
from sut_algorithms.highway_env.value_iteration import ValueIterationPolicy


def test_longitudinal_and_cutin_lane_counts_are_isolated():
    for mode, expected in (("lead_braking", 1), ("stop_and_go", 1),
                           ("slow_lead_following", 1),
                           ("fast_intrusion", 2), ("cutin_braking", 2)):
        scenario = CutInScenario(18.0, -4.0, mode, 0.5, 0.5)
        env = SingleLaneExternalCutInEnv(scenario)
        try:
            env.config["policy_frequency"] = 20
            env.reset(seed=711)
            assert env.config["lanes_count"] == expected
            assert len(env.road.network.graph["0"]["1"]) == expected
            assert env.ttc_observation().shape[1] == expected
        finally:
            env.close()


def test_one_lane_vi_and_fvdm_execute_without_changing_old_env():
    scenario = CutInScenario(18.0, -4.0, "lead_braking", 0.5, 0.5)
    env = SingleLaneExternalCutInEnv(scenario)
    policy = ValueIterationPolicy()
    try:
        env.config["policy_frequency"] = 20
        env.reset(seed=712)
        policy.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(policy.act(env))
        assert env.external_result().steps > 0
    finally:
        env.close()

    source = SingleLaneLongitudinalEnv(BUILDS["fvdm_ref"], scenario)
    try:
        source.reset(seed=712)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = source.step(1)
        assert source.steps > 0
    finally:
        source.close()
