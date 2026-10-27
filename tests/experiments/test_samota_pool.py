import numpy as np
import pytest
import torch

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.samota.pool import (
    Config,
    RadialNetwork,
    local_clusters,
    mixed_distance,
    run,
)


def test_trainable_radial_layer_matches_gaussian_centers_and_individual_widths():
    torch.manual_seed(23)
    x = np.array([[0, 0], [1, 1], [-1, 0]], dtype=np.float32)
    model = RadialNetwork(x, centers=3, seed=11)
    with torch.no_grad():
        model.betas.copy_(torch.tensor([1.0, 2.0, 3.0]))
        model.output.weight.fill_(1)
        model.output.bias.zero_()
        actual = model(torch.tensor(x)).numpy()
    centers = model.centers.detach().numpy()
    reference = np.exp(-np.array([1, 2, 3]) * ((x[:, None] - centers) ** 2).sum(-1)).sum(-1)
    assert actual == pytest.approx(reference, abs=1e-7)
    assert model.centers.requires_grad and model.betas.requires_grad


def test_mixed_distance_and_hdbscan_keep_distinct_promising_regions():
    rng = np.random.default_rng(11)
    x = np.zeros((20, 7))
    x[:10, :5] = rng.uniform(0, 0.01, (10, 5))
    x[10:, :5] = rng.uniform(0.99, 1, (10, 5))
    x[:10, 5], x[10:, 6] = 1, 1
    distance = mixed_distance(x, "cut_in")
    assert np.diag(distance) == pytest.approx(np.zeros(20))
    assert distance[0, 10] > 0.9
    clusters = local_clusters(x, np.zeros((20, 3)), 0, "cut_in", config=Config(elite_fraction=1))
    assert len(clusters) == 2
    assert all(np.all(indices < 10) or np.all(indices >= 10) for indices in clusters)


def test_fully_covered_objectives_restart_with_history_and_budget_preserved():
    scenes = sample("cut_in", 16, "development", replicate=43)

    def execute(scene):
        return PaidExecution(
            QueryFeedback(scene.scenario_id, True, SafetyMeasurements(0, 0), 1), []
        )

    ledger = QueryLedger(scenes, 12, execute)
    result = run_isolated(
        run,
        {
            "candidates": scenes,
            "initial_indices": [4, 1, 8, 3],
            "budget": 12,
            "config": Config(rbf_epochs=2),
        },
        ledger,
    )
    assert result["selected_indices"][:4] == [4, 1, 8, 3]
    assert len(set(result["selected_indices"])) == result["actual_queries"] == 12
    assert result["coverage_restarts"][0]["after_paid_queries"] == 4
    assert {row["reason"] for row in result["queries"]} >= {"global", "local"}
    assert sum(row["adaptation_s"] for row in result["queries"]) > 0
