from replications.scenariofuzz_highway_env.scenariofuzz.corpus import ScenarioSpec
from replications.scenariofuzz_highway_env.scenariofuzz.execution import TRACE_FIELDS, execute_scenario
from diva_highway_env.sut.idm_profiles import get_profile


def test_actual_highway_episode_records_both_vehicles(tmp_path):
    spec = ScenarioSpec.create(12, -6, "fast_intrusion")
    result = execute_scenario(get_profile("SUT-C"), spec, 9, tmp_path)
    assert result.valid
    assert result.trajectory_path is not None
    import numpy as np
    with np.load(result.trajectory_path, allow_pickle=False) as data:
        assert data["values"].shape[1] == len(TRACE_FIELDS)
        assert len(data["values"]) > 0

