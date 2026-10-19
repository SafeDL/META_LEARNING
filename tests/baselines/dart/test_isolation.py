"""Exercise the Windows spawn disclosure channel and report dead workers."""

import multiprocessing as mp

import numpy as np
import pytest

from methods.baselines.dart.evaluation.evaluate import worker
from benchmark.risk_channel import disclose_risk


def test_paid_risk_only_channel_under_spawn():
    size = 201
    x = np.zeros((size, 5))
    covariance = np.eye(size) * 0.01
    priors = np.full((size, 1), 0.8)
    task = {
        "policy": "cg_brt",
        "x": x,
        "covariance": covariance,
        "priors": priors,
        "probability": np.array([1.0]),
        "fixed_prior": priors[:, 0],
        "seed": 11,
    }
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=worker, args=(child,))
    process.start()
    child.close()
    try:
        result = disclose_risk(parent, task, np.full(size, 0.8), process)
        assert len(set(result["selected_indices"])) == 200
        assert len(result["disclosures"]) == 200
        assert all(q["collision"] is None for q in result["disclosures"])
        assert all(q["risk"] == 0.8 for q in result["queries"])
        parent.send(("stop", None))
        process.join(timeout=15)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join()


def test_dead_selector_is_reported():
    class Connection:
        def send(self, value):
            pass

        def poll(self, timeout):
            return False

    class Process:
        def is_alive(self):
            return False

    with pytest.raises(RuntimeError, match="exited"):
        disclose_risk(Connection(), {}, np.zeros(200), Process())
