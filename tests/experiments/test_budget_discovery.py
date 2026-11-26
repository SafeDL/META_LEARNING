"""E02 permissions, prescribed ranking and feedback-driven Pooled selection."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.budget import selectors
from experiments.formal_scenarios import map_scene
from experiments.rss import measurement_contract
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.query import PaidExecution
from experiments.resume import ReplayInterrupted, ResumableLedger
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.budget.report import paired_statistics
from testing_methods.baselines.scenario_fuzz.graphs import graph
from testing_methods.baselines.scenario_fuzz.model import ScenarioRiskNetwork, batch_graphs
from testing_methods.history_transfer.adaptation import FitSettings


@pytest.mark.parametrize("kind", FORMAL_SCENARIOS)
def test_formal_graph_public_geometry_and_batching(kind):
    torch.set_num_threads(1)
    scenes = [map_scene(kind, np.full(4, value), f"public_{i}")[0]
              for i, value in enumerate((0.1, 0.5, 0.9))]
    data = [graph(s) for s in scenes]
    assert torch.equal(data[0].x, graph(replace(scenes[0], scenario_id="hidden_ads")).x)
    for item in data:
        assert torch.isfinite(item.x).all()
        assert item.edge_index.max() < len(item.x)
        assert item.edge_graph_index.max() < len(item.edge_attr)
        road_y = set(item.x[item.x[:, 3] == 0, 1].tolist())
        assert len(road_y) == (1 if kind == FORMAL_SCENARIOS[0] else 3 if kind == "cut_in_onset" else 2)
    model = ScenarioRiskNetwork(hidden_dim=4, heads=1).eval()
    model.fit_normalization(data)
    with torch.no_grad():
        alone = torch.cat([model(batch_graphs([d])) for d in data])
        together = model(batch_graphs(data))
    np.testing.assert_allclose(alone, together, atol=1e-6)


def test_static_order_follows_same_greedy_ties_and_is_feedback_independent():
    scenes = [SimpleNamespace(scenario_id=i) for i in ("z", "b", "a", "c")]
    probability = [0.8, 0.7, 0.7 - 5e-13, 0.4]
    assert selectors.static_order(scenes, probability, [3], 4) == [3, 0, 2, 1]


def test_pooled_selection_fits_only_its_paid_prefix(monkeypatch):
    scenes = [map_scene("car_following_braking_onset", np.full(4, i / 5), f"public_{i}")[0]
              for i in range(5)]
    fit_prefixes = []

    def predict(base, x, labels, margins, query_x, **kwargs):
        fit_prefixes.append(list(labels))
        assert len(x) == len(labels) == len(margins)
        return np.linspace(0.1, 0.9, len(query_x)), {}, {}

    monkeypatch.setattr(selectors, "predict_pooled", predict)
    monkeypatch.setattr(selectors, "synchronized_time", lambda device: 0.0)
    observed = []

    class Oracle:
        def query(self, index, *, purpose):
            assert index not in observed
            observed.append(index)
            return SimpleNamespace(
                scenario_id=scenes[index].scenario_id, collision=index % 2 == 0,
                safety=SimpleNamespace(rss_margin=1.0), rss_contract=measurement_contract(),
                to_dict=lambda: {"collision": index % 2 == 0})

    result = selectors.pooled_select(
        scenes, Oracle(), pretrained=torch.nn.Linear(4, 1),
        history={"x": np.zeros((2, 4)), "collision": [False, True], "rss_margin": [1, -1]},
        initial_indices=[0, 1], budget=4, fit_settings=FitSettings(10, 0.003), snapshot_counts=(0, 2), device="cpu")
    assert [len(p) for p in fit_prefixes] == [2, 3, 4, 5]
    assert result["selected_indices"] == [0, 1, 4, 3]
    assert [q["conditioning_queries"] for q in result["queries"]] == [0, 1, 2, 3]


def test_static_recovery_reuses_paid_prefix_without_changing_order(tmp_path):
    scenes = [map_scene("car_following_braking_onset", np.full(4, i / 5), f"public_{i}")[0]
              for i in range(5)]
    paid_calls = []

    def execute(scene):
        paid_calls.append(scene.scenario_id)
        return PaidExecution(QueryFeedback(scene.scenario_id, False, SafetyMeasurements(0, 0), 1), [])

    arguments = {"checkpoint_path": tmp_path / "paid.json.gz", "specification": {"unit": "static"},
                 "paid_execution": lambda i: PaidExecution(
                     QueryFeedback(scenes[i].scenario_id, False, SafetyMeasurements(0, 0), 1), [])}
    public = {"method": "static_prior", "initial_indices": [2], "budget": 4,
              "probabilities": [0.2, 0.1, 0.5, 0.8, 0.9]}
    ledger = ResumableLedger(scenes, 4, execute, stop_after=2, **arguments)
    with pytest.raises(ReplayInterrupted):
        selectors.run(scenes, ledger, **public)
    resumed = ResumableLedger(scenes, 4, execute, **arguments)
    result = selectors.run(scenes, resumed, **public)
    resumed.finish()
    assert result["selected_indices"] == [2, 4, 3, 0]
    assert len(paid_calls) == 4


def test_discovery_statistics_resample_ads_after_equal_scene_averages():
    values = np.zeros((8, 4, 7, 2))
    values[:, :, 0, 0] = np.arange(8)[:, None] + np.arange(4)[None, :]
    values[:, :, 0, 1] = 2 * values[:, :, 0, 0]
    summary = paired_statistics(values)
    paired = summary["scopes"]["macro"]["ours_minus"]["target_only"]
    assert paired["gain"] == [5.0, 10.0]
    assert len(paired["ads_gains"]) == 8
    assert paired["both_metrics_supported"]
    assert paired["simultaneous"]["10"]["interval_coverage"] == 0.995
    assert summary["scopes"]["macro"]["methods"]["random"]["C200"] == 0
