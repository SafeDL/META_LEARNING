import numpy as np
import torch

from experiments.query import PaidExecution, QueryLedger
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.response_baselines.collision import run
from testing_methods.common.gp_classifier import (
    ClassificationKernel,
    collision_probabilities,
    fit_target,
    laplace,
)


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


def test_random_baseline_shares_paid_initialization_and_fixed_budget():
    scenes = sample("front_brake", 12, "new")

    def execute(scenario):
        return PaidExecution(
            QueryFeedback(scenario.scenario_id, False, SafetyMeasurements(30, 8), 1), []
        )

    ledger = QueryLedger(scenes, 6, execute)
    result = run(scenes, ledger, method="random", initial_indices=[1, 3], budget=6)
    assert result["selected_indices"][:2] == [1, 3]
    assert len(set(result["selected_indices"])) == 6
    assert len(ledger.records) == result["actual_queries"] == 6
