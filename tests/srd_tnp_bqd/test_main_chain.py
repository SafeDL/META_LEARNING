"""Replay the current mean-readout main chain against its measured trajectory."""
from dataclasses import asdict
import multiprocessing as mp

import numpy as np
import pytest
import torch

from methods.srd_tnp_bqd.common import read_json
from methods.srd_tnp_bqd.experiment import worker
from methods.srd_tnp_bqd.benchmark import BUDGET
from methods.srd_tnp_bqd.s01 import ROOT, cells_for, numeric_inputs, response_rows
from methods.srd_tnp_bqd.oracle import ContinuousOracle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="frozen historical model requires CUDA")
def test_frozen_main_chain_replays_existing_200_queries():
    task = {"method": "SRD_TNP_BQD", "seed": 11}
    reference = read_json(ROOT / "comparison/runs/SRD_TNP_BQD_seed_11.json")
    scenes, rows = response_rows("D")
    oracle = ContinuousOracle([s["scenario_id"] for s in scenes],
                              lambda i: (int(rows[i]["ego_collision"]), rows[i]["risk"], rows[i]["valid_risk"]),
                              BUDGET)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker, args=(child, task, numeric_inputs(scenes), cells_for(scenes)))
    process.start()
    child.close()
    records = []
    try:
        while True:
            assert parent.poll(60), "selector stopped responding"
            message = parent.recv()
            if message["type"] == "query":
                parent.send({"observation": asdict(oracle.query(message["index"]))})
            elif message["type"] == "record":
                records.append(message["row"])
            elif message["type"] == "complete":
                result = message["result"]
                break
            else:
                pytest.fail(message["message"])
    finally:
        parent.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    assert len(records) == len(oracle.queried) == BUDGET
    assert result["selected_indices"] == reference["selected_indices"]
    assert result["frozen_modules"] and result["frozen_history_cache"]
    for row, saved in zip(result["queries"], reference["queries"]):
        assert row["index"] == saved["index"] and row["risk"] == saved["risk"] and row["label"] == saved["label"]
        for key in ("m", "m0", "mu", "latent_var", "EAI", "posterior_risk_mean", "e",
                    "mean_offset", "mean_scale", "mean_offset_sd", "mean_scale_sd"):
            np.testing.assert_allclose(row[key], saved[key], rtol=0, atol=1e-10)
    for key in ("final_mean", "final_latent_var", "final_risk_mean"):
        np.testing.assert_allclose(result[key], reference[key], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result["mean_calibration"]["coefficient_mean"],
                               reference["mean_calibration"]["coefficient_mean"], rtol=0, atol=1e-10)
    assert result["mean_readout"]["D_used"] is False
