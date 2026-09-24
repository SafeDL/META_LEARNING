from types import SimpleNamespace

import numpy as np
import pytest

from method_chains.core_mine.hierarchical_residual import HierarchicalResidualModel


def _task():
    return SimpleNamespace(
        source_y=np.asarray([[0.10, 0.20, 0.10],
                             [0.20, 0.30, 0.20]]),
        features=np.asarray([[0.0], [0.9], [0.0]]),
        modes=np.asarray(["a", "a", "b"]),
        count=3,
    )


def test_global_shift_reaches_far_point_but_not_other_mode():
    task = _task()
    model = HierarchicalResidualModel(task)
    prior = model.predict()
    np.testing.assert_allclose(prior["mean"], task.source_y.mean(axis=0))
    model.observe(0, 1.0)
    after = model.predict()
    assert after["mean"][0] > prior["mean"][0]
    assert after["mean"][1] > prior["mean"][1]
    assert after["mean"][1] < after["mean"][0]
    assert after["mean"][2] == prior["mean"][2]
    assert np.all(after["p_collision"] <= after["p_event"])
    assert np.all((0 <= after["p_event"]) & (after["p_event"] <= 1))


def test_revealed_response_is_unique_and_finite():
    model = HierarchicalResidualModel(_task())
    model.observe(0, 0.8)
    with pytest.raises(ValueError):
        model.observe(0, 0.9)
    with pytest.raises(ValueError):
        model.observe(1, float("nan"))
