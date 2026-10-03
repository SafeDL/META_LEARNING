import numpy as np

from methods.srd_tnp_bqd.s01 import (COUNT, SOURCES, TARGET, build_spec, cells_for,
                                   load_scenes, numeric_inputs, response_rows)


def test_original_s01_manifests_sources_target_and_ground_truth():
    a, d = load_scenes("A"), load_scenes("D")
    assert len(a) == len(d) == COUNT == 2048
    assert len(SOURCES) == 5 and all(build_spec(name).profile["controller"] == "IDM" for name in SOURCES)
    assert build_spec(TARGET).profile["target_speed"] == 23.
    assert build_spec(TARGET).profile["max_brake"] == 8.
    xa, xd = numeric_inputs(a), numeric_inputs(d)
    assert xa.shape == xd.shape == (2048, 4)
    assert not set(map(tuple, xa)) & set(map(tuple, xd))
    assert all(s["template_id"] == "fbrt_cutin" and s["fixed_context"]["ego_speed_mps"] == 25 for s in a + d)
    _, rows = response_rows("D")
    labels = np.array([r["ego_collision"] for r in rows])
    assert labels.sum() == 116 and len(set(cells_for(d)[labels])) == 33
