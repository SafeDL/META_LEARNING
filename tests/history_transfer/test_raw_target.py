import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.history_transfer.inference import adapted_model, predict_components
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.selection import select


class Oracle:
    def __init__(self, scenes, margins):
        self.scenes = scenes
        self.margins = margins

    def query(self, index, *, purpose):
        return QueryFeedback(
            self.scenes[index].scenario_id,
            index == 1,
            SafetyMeasurements(1, 1, rss_margin=self.margins[index]),
            0,
            measurement_contract(),
        )


def test_prediction_snapshots_use_exact_paid_prefix_and_include_the_final_adapted_state():
    torch.set_num_threads(1)
    scenes = sample("car_following", 8, "new", replicate=90)
    margins = np.linspace(-0.8, 0.6, len(scenes))
    result = select(
        scenes, Oracle(scenes, margins), initial_indices=[0, 1], budget=3, snapshot_counts=(0, 2, 3)
    )
    snapshots = result["prediction_snapshots"]
    assert [s["paid_queries"] for s in snapshots] == [0, 2, 3]
    assert [len(s["observed_indices"]) for s in snapshots] == [0, 2, 3]
    assert result["actual_queries"] == 3
    assert result["diagnostic_prediction_s"] > 0
    x = features(scenes)
    for snapshot in snapshots:
        model = JointResponseKernel(3, deep=False).double()
        with torch.no_grad():
            for name, value in snapshot["target_parameters"].items():
                getattr(model, name).fill_(value)
            paid = snapshot["observed_indices"]
            reference, _ = predict_components(
                model,
                None,
                x,
                [],
                model.point_mean(1)[0],
                paid,
                [i == 1 for i in paid],
                margins[paid],
            )
        np.testing.assert_allclose(snapshot["probabilities"], reference[0], rtol=1e-12, atol=1e-12)
    changed = select(
        scenes,
        Oracle(scenes, margins + 0.2),
        initial_indices=[0, 1],
        budget=3,
        snapshot_counts=(0, 2, 3),
    )
    assert snapshots[0] == changed["prediction_snapshots"][0]
    assert snapshots[-1]["probabilities"] != changed["prediction_snapshots"][-1]["probabilities"]


def test_raw_control_constructs_no_deep_model_and_matches_single_joint_gp(monkeypatch):
    torch.set_num_threads(1)
    scenes = sample("car_following", 8, "new", replicate=90)
    margins = np.linspace(-0.8, 0.6, len(scenes))
    construction = JointResponseKernel.__init__
    deep_flags = []

    def checked(self, dimension, *, deep=True):
        deep_flags.append(deep)
        construction(self, dimension, deep=deep)

    monkeypatch.setattr(JointResponseKernel, "__init__", checked)
    result = select(scenes, Oracle(scenes, margins), initial_indices=[0, 1, 2], budget=3)
    assert deep_flags == [False]
    assert result["component_names"] == ["raw_target"]
    assert not result["explicit_source_components"] and not result["global_reliability"]
    assert result["physical_mass"] == 1
    assert result["representation"] == "normalized_parameters"
    assert all(q["prequery_component_weights"] == [1] for q in result["queries"])
    assert all(q["prequery_log_likelihood"] is None for q in result["queries"])

    base = JointResponseKernel(3, deep=False).double()
    x = features(scenes)
    model = adapted_model(base, x[:2], [False, True], margins[:2], learning_rate=0.01)
    with torch.no_grad():
        probabilities, diagnostics = predict_components(
            model,
            None,
            x,
            [],
            base.point_mean(1)[0],
            [0, 1],
            [False, True],
            margins[:2],
            batch_size=2,
        )
    assert probabilities.shape == (1, len(scenes)) and len(diagnostics) == 1
    assert result["queries"][2]["prequery_probability"] == pytest.approx(
        float(probabilities[0, 2]), abs=1e-12
    )


def test_raw_control_uses_paid_rss_without_random_features_or_future_feedback():
    scenes = sample("car_following", 8, "new", replicate=91)
    margins = np.linspace(-0.8, 0.6, len(scenes))

    def run(seed, rss, batch_size):
        torch.manual_seed(seed)
        return select(
            scenes, Oracle(scenes, rss), initial_indices=[0, 1], budget=4, batch_size=batch_size
        )

    first, second = run(1, margins, 1), run(999, margins, 128)
    assert first["selected_indices"] == second["selected_indices"]
    np.testing.assert_allclose(
        [q["prequery_probability"] for q in first["queries"]],
        [q["prequery_probability"] for q in second["queries"]],
        atol=1e-12,
    )
    changed = margins.copy()
    changed[0] = 0.9
    altered = run(1, changed, 1)
    assert (
        altered["queries"][0]["prequery_probability"] == first["queries"][0]["prequery_probability"]
    )
    assert (
        abs(
            altered["queries"][1]["prequery_probability"]
            - first["queries"][1]["prequery_probability"]
        )
        > 1e-6
    )
    unseen = next(i for i in range(len(scenes)) if i not in first["selected_indices"])
    changed = margins.copy()
    changed[unseen] = 0.9
    hidden = run(1, changed, 7)
    assert hidden["selected_indices"] == first["selected_indices"]


def test_raw_control_rejects_historical_source_data():
    with pytest.raises(ValueError, match="cannot use historical"):
        select(sample("car_following", 3, "new"), None, sources={}, initial_indices=[0], budget=2)
