import numpy as np

from methods.baselines.hem.experts import HistoricalExperts, neighbor_prediction
from methods.baselines.hgt.history import historical_risk, load_history, split_indices, subset


def test_real_bank_merge_exactly_preserves_legacy_prior():
    history = load_history()
    x = next(iter(history.values()))["x"]
    train, validation = split_indices(x)
    history = subset(history, train)
    experts = HistoricalExperts(history).predict(x[validation])
    np.testing.assert_array_equal(experts.fixed_mean(), historical_risk(history, x[validation]))


def test_neighbor_search_isolates_scene_families():
    source = {"x": np.r_[np.column_stack([np.full((10, 4), .9), np.zeros(10)]),
                        np.column_stack([np.zeros((10, 4)), np.ones(10)])],
              "risk": np.r_[np.ones(10), np.zeros(10)]}
    mean = neighbor_prediction(source, np.array([[0, 0, 0, 0, 0]]))
    np.testing.assert_allclose(mean, [1])


def test_experts_allow_different_coordinate_banks():
    rng = np.random.default_rng(11)
    history = {name: {"x": np.column_stack([rng.random((count, 4)), np.zeros(count)]),
                      "risk": rng.random(count)} for name, count in [("idm_reactive", 20), ("idm_limited", 25)]}
    query = np.column_stack([rng.random((5, 4)), np.zeros(5)])
    predictions = HistoricalExperts(history).predict(query)
    assert predictions.mean.shape == (5, 2)
    assert np.isfinite(predictions.mean).all()

