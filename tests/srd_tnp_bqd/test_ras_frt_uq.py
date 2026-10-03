import pytest

from methods.ras_frt_uq import s01 as ras
from methods.srd_tnp_bqd.common import read_json
from methods.srd_tnp_bqd.oracle import ContinuousOracle
from methods.srd_tnp_bqd.s01 import cells_for, numeric_inputs, response_rows


@pytest.mark.parametrize("seed", [11, 23, 37, 53, 71])
def test_original_ras_200_query_trajectory_is_preserved(seed):
    scenes, rows = response_rows("D")
    oracle = ContinuousOracle([s["scenario_id"] for s in scenes],
                              lambda i: (int(rows[i]["ego_collision"]), rows[i]["risk"], True), 200)
    result = ras.select(numeric_inputs(scenes), cells_for(scenes), oracle, seed)
    reference = next(r for r in read_json(ras.MODEL_ROOT / "original_replay.json")["runs"] if r["seed"] == seed)
    assert result["selected_indices"] == reference["selected_indices"]
    assert len(set(oracle.queried)) == 200
    assert result["feedback_used"] == "binary collision"
