from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.indago.method import FailurePredictor, TrainingConfig, load, train
from testing_methods.baselines.indago.pool import PoolSearch, SearchConfig, run
from testing_methods.common.response import ResponseArchive


@pytest.mark.parametrize("label", (False, True))
def test_single_historical_class_keeps_finite_two_logit_training_without_synthetic_labels(
    tmp_path, label
):
    scenes = sample("front_brake", 64, "history")
    archive = ResponseArchive(
        "history",
        "rule",
        "idm",
        "history",
        scenes,
        np.full(64, label, dtype=bool),
        np.ones((64, 2)),
    )
    path = tmp_path / "single_class.pt"
    report = train([archive], path, config=TrainingConfig(epochs=2))
    model, _ = load(path)
    expected = [0.0, 0.0]
    expected[int(label)] = 1.0
    assert report["class_weights"] == expected
    assert report["historical_training_class_counts"][int(not label)] == 0
    assert torch.isfinite(model(features(scenes))).all()
    assert model(features(scenes)).shape == (64, 2)


def test_indago_saliency_samples_absolute_failure_logit_gradient_and_keeps_its_sign():
    scenes = sample("front_brake", 8, "development")
    model = FailurePredictor(6, layers=1)
    model.network = torch.nn.Sequential(torch.nn.Linear(6, 2))
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].weight[1, 1] = -9
        model.network[0].weight[1, 2] = 1
        model.network[0].bias.zero_()
    search = PoolSearch(scenes, model)
    draws = [search.saliency(0) for _ in range(1000)]
    fraction = sum(coordinate == 1 for coordinate, _ in draws) / len(draws)
    assert 0.85 < fraction < 0.95
    assert {coordinate for coordinate, _ in draws} == {1, 2}
    assert all(
        direction < 0 if coordinate == 1 else direction > 0 for coordinate, direction in draws
    )


def test_pool_projection_is_public_and_excludes_paid_coordinates():
    base = sample("front_brake", 1, "development")[0]
    scenes = [
        replace(base, scenario_id=str(i), parameters={**base.parameters, "gap_m": gap})
        for i, gap in enumerate((30, 31, 32))
    ]
    search = PoolSearch(scenes, FailurePredictor(6, layers=1))
    assert search.project(scenes[1]) == 1
    search.available.remove(1)
    assert search.project(scenes[1]) in {0, 2}


def test_historical_training_has_its_own_weighted_classifier_and_rejects_target_data(tmp_path):
    scenes = sample("front_brake", 32, "history")
    archive = ResponseArchive(
        "history", "rule", "idm", "history", scenes, np.arange(32) % 4 == 0, np.ones((32, 2))
    )
    path = tmp_path / "indago.pt"
    report = train([archive], path, config=TrainingConfig(layers=1, epochs=2))
    model, metadata = load(path)
    assert metadata["history_ids"] == ["history"]
    assert report["training_rows"] + report["validation_rows"] == 32
    assert report["class_weights"][1] > report["class_weights"][0]
    probabilities = model.probabilities(features(scenes))
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    with pytest.raises(ValueError, match="historical objects only"):
        train([replace(archive, role="development")], tmp_path / "leaked.pt")
    with pytest.raises(ValueError, match="Held family"):
        train([archive], tmp_path / "held.pt", held_family="rule")


def test_fixed_classifier_sequence_ignores_target_labels_and_pays_full_initialization():
    scenes = sample("front_brake", 16, "development")
    torch.manual_seed(11)
    classifier = FailurePredictor(6, layers=1).eval()
    before = {k: v.clone() for k, v in classifier.state_dict().items()}
    results = []
    for label in (False, True):

        def execute(scene):
            return PaidExecution(
                QueryFeedback(scene.scenario_id, label, SafetyMeasurements(1, 1), 1), []
            )

        ledger = QueryLedger(scenes, 8, execute)
        result = run_isolated(
            run,
            {
                "candidates": scenes,
                "classifier": classifier,
                "initial_indices": [6, 3, 1, 9],
                "budget": 8,
                "config": SearchConfig(neighborhood_size=5, max_neighborhoods=3, max_stagnation=2),
            },
            ledger,
        )
        assert result["selected_indices"][:4] == [6, 3, 1, 9]
        assert len(set(result["selected_indices"])) == result["actual_queries"] == 8
        assert all(row["adaptation_s"] == 0 for row in result["queries"])
        results.append(result)
    assert results[0]["selected_indices"] == results[1]["selected_indices"]
    assert all(torch.equal(before[k], v) for k, v in classifier.state_dict().items())
