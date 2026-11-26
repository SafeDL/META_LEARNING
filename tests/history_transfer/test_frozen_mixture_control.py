"""Under fixed feedback, freezing mixture weights must leave both target GP fits identical."""

import numpy as np
import pytest
import torch

from experiments.history import build_source
from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer import selection
from testing_methods.history_transfer.adaptation import FitSettings, OnlineSettings
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_joint_feedback_weight_control_changes_only_mixture_weights(tmp_path):
    torch.set_num_threads(1)
    scenes = sample("car_following", 6, "new", replicate=902)
    margins = np.linspace(0.7, -0.5, 6)
    history = [
        ResponseArchive(
            f"source_{label}",
            "rule",
            f"rule_{label}",
            "history",
            scenes,
            np.full(6, bool(label)),
            np.ones((6, 2)),
            rss_margin=margins * (1 if label == 0 else -1),
            rss_contract=measurement_contract(),
        )
        for label in (0, 1)
    ]
    model = SharedPrivateKernel(features(scenes).shape[1], deep=False).double()
    sources = build_source(model, history, scenes, cache_path=tmp_path / "source.npz")

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                index == 2,
                SafetyMeasurements(1, 1, rss_margin=float(margins[index])),
                0,
                measurement_contract(),
            )

    settings = OnlineSettings(FitSettings(3, 0.01), FitSettings(3, 0.003), -1.4)
    common = dict(
        pretrained=model,
        sources=sources,
        initial_indices=[0],
        budget=4,
        query_order=[0, 1, 2, 3],
        snapshot_counts=(0, 2, 4),
        online_settings=settings,
    )
    dynamic = select(scenes, Oracle(), update_source_weights=True, **common)
    frozen = select(scenes, Oracle(), update_source_weights=False, **common)
    for d, f in zip(dynamic["prediction_snapshots"], frozen["prediction_snapshots"]):
        assert d["observed_indices"] == f["observed_indices"]
        assert d["target_parameters"] == f["target_parameters"]
        assert d["physical_parameters"] == f["physical_parameters"]
        assert d["target_fit"] == f["target_fit"] and d["physical_fit"] == f["physical_fit"]
        assert f["component_weights"] == frozen["prediction_snapshots"][0]["component_weights"]
        assert f["component_weights"][-1] == pytest.approx(0.1)
    assert dynamic["prediction_snapshots"][-1]["component_weights"][-1] != pytest.approx(0.1)
    assert (
        dynamic["prediction_snapshots"][-1]["component_weights"]
        != dynamic["prediction_snapshots"][0]["component_weights"]
    )
    assert (
        dynamic["prediction_snapshots"][-1]["probabilities"]
        != frozen["prediction_snapshots"][-1]["probabilities"]
    )
    assert dynamic["target_feedback_updates"] and frozen["target_feedback_updates"]
    assert dynamic["feedback_used"] == frozen["feedback_used"] == "joint"
    assert dynamic["source_weight_updates"] and not frozen["source_weight_updates"]


@pytest.mark.parametrize("collision", [False, True])
def test_paid_feedback_can_favour_or_reject_the_raw_target_component(monkeypatch, collision):
    torch.set_num_threads(1)
    scenes = sample("car_following", 6, "new", replicate=903)
    model = SharedPrivateKernel(features(scenes).shape[1], deep=False).double()
    sources = {
        "names": ["history"],
        "weights": np.ones(1),
        "means": np.zeros((1, len(scenes), 2)),
        "blocks": np.tile(np.eye(2), (1, len(scenes), 1, 1)),
    }

    def predictions(target, physical, x, *args, indices=None, **kwargs):
        count = len(x) if indices is None else len(indices)
        probabilities = torch.tensor([0.9, 0.8, 0.01], dtype=torch.float64)
        return probabilities[:, None].expand(-1, count), [{}, {}, {}]

    monkeypatch.setattr(selection, "predict_components", predictions)

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                collision,
                SafetyMeasurements(1, 1, rss_margin=-0.5 if collision else 0.5),
                0,
                measurement_contract(),
            )

    result = select(
        scenes,
        Oracle(),
        pretrained=model,
        sources=sources,
        initial_indices=[0],
        query_order=[0, 1, 2, 3],
        budget=4,
        snapshot_counts=(0, 4),
    )
    initial, final = result["prediction_snapshots"]
    assert initial["component_weights"] == pytest.approx([0.675, 0.225, 0.1])
    raw_mass = final["component_weights"][-1]
    assert raw_mass < 0.001 if collision else raw_mass > 0.99
    assert result["physical_mass"] == pytest.approx(raw_mass)
    assert all(len(q["prequery_log_likelihood"]) == 3 for q in result["queries"])
    expected = np.asarray(final["component_weights"]) @ np.array([0.9, 0.8, 0.01])
    np.testing.assert_allclose(final["probabilities"], expected, rtol=0, atol=1e-12)
