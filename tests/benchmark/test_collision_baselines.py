from dataclasses import replace

import numpy as np
import pytest
import torch

from methods.baselines.response_baselines.collision import run
from benchmark.query import PaidExecution, QueryLedger
from benchmark.scenarios import sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.baselines.fsbo.method import CONFIG as FSBO_CONFIG
from methods.baselines.fsbo.method import load as load_fsbo
from methods.baselines.fsbo.method import train as train_fsbo
from methods.baselines.pooled_ft.method import CONFIG as POOLED_CONFIG
from methods.baselines.pooled_ft.method import train as train_pooled
from methods.baselines.pooled_ft.method import load as load_pooled
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target, laplace
from methods.common.response import ResponseArchive


def archive(name, role="history", *, replicate=0):
    scenes = sample("front_brake", 48, "history" if role == "history" else "development", replicate=replicate)
    return ResponseArchive(name, "rule", name, role, scenes, np.arange(48) % 3 == 0,
                           np.column_stack((np.linspace(0, 80, 48), np.ones(48))))


def test_laplace_updates_are_bernoulli_and_differentiable():
    torch.set_num_threads(1)
    x = np.linspace(0, 1, 8)[:, None]
    model = ClassificationKernel(1, deep=True)
    nll = laplace(model, x, [False] * 8)[0]
    nll.backward()
    assert model.embedding[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        p = collision_probabilities(model, x, [False] * 8, x).numpy()
        q = collision_probabilities(model, x, [True] * 8, x).numpy()
    assert np.all(p < 0.5) and np.all(q > 0.5)
    fit_target(model, x, [False] * 8, steps=2)
    assert torch.isfinite(model.mean)


@pytest.mark.parametrize("method", ["random", "space_filling", "history_ranking", "nearest_residual", "gp_collision"])
def test_pool_baselines_share_paid_initialization_and_fixed_budget(method):
    scenes = sample("front_brake", 12, "new")
    history = (archive("a"), archive("b", replicate=1))
    def execute(scenario):
        return PaidExecution(QueryFeedback(scenario.scenario_id, False,
                                           SafetyMeasurements(30, 8), 1), [])
    ledger = QueryLedger(scenes, 6, execute)
    result = run(scenes, ledger, method=method, initial_indices=[1, 3], budget=6,
                 history=history, adaptation_steps=1)
    assert result["selected_indices"][:2] == [1, 3]
    assert len(set(result["selected_indices"])) == 6
    assert len(ledger.records) == result["actual_queries"] == 6
    if method == "nearest_residual":
        assert len(result["queries"][0]["source_names"]) == 2
        assert len(result["queries"][1]["source_names"]) == 1


@pytest.mark.parametrize("method", ["fsbo", "pooled_ft"])
def test_learned_baselines_build_from_new_role_separated_archives(tmp_path, method):
    history = [archive("a"), archive("b", replicate=1)]
    development = [archive("dev", "development", replicate=2)]
    configuration = dict(FSBO_CONFIG if method == "fsbo" else POOLED_CONFIG)
    configuration.update(steps=2, validation_interval=1, batch_size=12,
                         support_count=4, query_count=8, adaptation_steps=1)
    destination = tmp_path / f"{method}.pt"
    trainer = train_fsbo if method == "fsbo" else train_pooled
    trainer(history, development, destination, config=configuration)
    scenes = sample("front_brake", 8, "new")
    def execute(scenario):
        return PaidExecution(QueryFeedback(scenario.scenario_id, True,
                                           SafetyMeasurements(0, 0), 1), [])
    ledger = QueryLedger(scenes, 4, execute)
    pretrained, metadata = (load_fsbo if method == "fsbo" else load_pooled)(destination)
    result = run(scenes, ledger, method=method, initial_indices=[0, 1], budget=4,
                 pretrained=pretrained, adaptation_config=metadata["config"], adaptation_steps=1)
    assert result["actual_queries"] == 4
    assert all(0 <= q["prequery_probability"] <= 1 for q in result["queries"])
    if method == "fsbo":
        _, metadata = load_fsbo(destination)
        assert metadata["history_ids"] == ["a", "b"]
    with pytest.raises(ValueError, match="Held family"):
        trainer(history, development, tmp_path / "held.pt", held_family="rule", config=configuration)
