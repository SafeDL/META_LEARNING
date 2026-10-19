"""Count distributions, prefix preservation, paid stopping and terminal costs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import betabinom

from benchmark.risk_channel import disclose_risk
from benchmark.timing import QueryClock
from methods.baselines.lrt.adaptive.config import horizons
from methods.baselines.lrt.adaptive.controller import StopController
from methods.baselines.lrt.adaptive.model import RemainingDiscoveries, training_loss
from methods.baselines.lrt.adaptive.train import stopped_result
from methods.common.counts import count_cdf, count_log_prob, expected_count
from methods.baselines.lrt.selection import select
from methods.baselines.lrt.value import ContinuationValue
from tests.baselines.lrt.test_method import Oracle, problem


@pytest.mark.parametrize("length", (1, 7, 160, 200))
def test_count_distribution_matches_scipy_and_normalizes(length):
    parameters = torch.tensor([0.7, 1.2, 3.4], dtype=torch.float64)
    k = torch.arange(length + 1, dtype=torch.float64)
    actual = count_log_prob(parameters, length, k).exp().numpy()
    zero = torch.sigmoid(parameters[0]).item()
    expected = (1 - zero) * betabinom.pmf(k.numpy(), length, 1.2, 3.4)
    expected[0] += zero
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert actual.sum() == pytest.approx(1)
    assert expected_count(parameters, length).item() == pytest.approx(
        float(k.numpy() @ expected)
    )
    for limit in (-1, 0, min(2, length), length, length + 1):
        assert count_cdf(parameters, length, limit).item() == pytest.approx(
            expected[: limit + 1].sum() if limit >= 0 else 0
        )


def test_count_training_has_finite_gradients_and_frozen_representations():
    network = RemainingDiscoveries(ContinuationValue())
    x = torch.randn(3, 3, 136)
    parameters = network(x)
    loss = training_loss(parameters, [1, 5, 20], [0, 1, 3])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.head.parameters())
    assert all(
        p.grad is None and not p.requires_grad for p in network.source.parameters()
    )
    assert all(
        p.grad is None and not p.requires_grad for p in network.point.parameters()
    )


def test_horizons_and_retention_threshold_use_integer_discoveries():
    assert horizons(7) == [1, 5, 7]
    assert horizons(1) == [1]
    for found in range(200):
        for more in range(201):
            assert (more <= found // 9) == (10 * found >= 9 * (found + more))


@pytest.mark.parametrize("stop_at", (3, 7, 20))
def test_stopping_preserves_policy_prefix_and_loss_charges(stop_at):
    common, prior, truth = problem()

    class Controller:
        def evaluate(self, models, remaining, anchor, queries, evidence, budget):
            assert len(queries) == len(evidence)
            return {"stop": len(queries) >= stop_at, "query": len(queries)}

    args = dict(
        **common,
        transfer_prior=prior,
        value_model=None,
        variant="no_value_learning",
        budget=12,
        initial_queries=3
    )
    full = select(**args, oracle=Oracle(truth))
    clock = QueryClock()
    stopped = select(**args, oracle=Oracle(truth), stopper=Controller(), clock=clock)
    n = min(stop_at, 12)
    assert stopped["selected_indices"] == full["selected_indices"][:n]
    assert stopped["queries"] == full["queries"][:n]
    assert stopped["decisions"] == full["decisions"][:n]
    assert stopped["loss_budget_initial"] == pytest.approx(1.2)
    assert len(clock.report()["query_compute_s"]) == n
    if n < 12:
        assert clock.report()["final_decision_s"] > 0
        assert stopped["stop_reason"] == "quality_retained"
    else:
        assert stopped["stop_reason"] == "budget_exhausted"


def test_stop_requires_agreement_and_can_handle_zero_discoveries():
    class Network:
        disagree = True

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def features(self, *args):
            return None

        def __call__(self, features):
            output = torch.tensor(
                [[[10.0, 1.0, 1.0]], [[10.0, 1.0, 1.0]], [[-10.0, 100.0, 1.0]]]
            )
            if not self.disagree:
                output[2] = output[0]
            return output

    controller = StopController(Network(), 0.9)
    result = controller.evaluate(None, None, None, [], [], 200)
    assert result["allowed_additional"] == 0
    assert not result["stop"]
    assert result["model_retention_probability"][0] > 0.99
    assert result["model_retention_probability"][2] < 0.01
    controller.network.disagree = False
    assert controller.evaluate(None, None, None, [], [], 200)["stop"]


def test_terminal_prediction_time_is_included_without_an_extra_query(monkeypatch):
    import benchmark.timing as timing

    times = iter([1.0, 2.0, 5.0, 9.0])
    monkeypatch.setattr(timing, "now", lambda: next(times))
    clock = QueryClock()
    clock.start()
    clock.step()
    clock.finish_decision()
    report = clock.report()
    assert report["task_initialization_s"] == 1
    assert report["query_compute_s"] == [3]
    assert report["final_decision_s"] == 4


def test_early_oracle_return_must_be_explicitly_enabled_and_declared():
    risk = [0.1, 0.9, 0.8]
    result = {
        "selected_indices": [0, 1, 2],
        "queries": [{"index": i, "risk": r} for i, r in enumerate(risk)],
        "stop_reason": "quality_retained",
        "stopping": [{"stop": True}],
    }

    class Connection:
        def __init__(self):
            self.messages = iter(
                [*(("query", i) for i in range(3)), ("result", result)]
            )

        def send(self, value):
            pass

        def poll(self, timeout):
            return True

        def recv(self):
            return next(self.messages)

    process = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(ValueError, match="incomplete"):
        disclose_risk(Connection(), {}, risk, process)
    accepted = disclose_risk(Connection(), {}, risk, process, minimum_queries=3)
    assert len(accepted["disclosures"]) == 3
    assert all(r["collision"] is None for r in accepted["disclosures"])
    result["stopping"] = []
    with pytest.raises(ValueError, match="explicit stopping"):
        disclose_risk(Connection(), {}, risk, process, minimum_queries=3)


def test_stopping_replay_never_pads_with_unexecuted_discoveries():
    task = {"budget": 20, "positives": 10, "found": [min(t, 10) for t in range(1, 21)]}
    forecast = [
        {
            "query": 10,
            "found": 10,
            "probability": [0.99] * 3,
            "mean": [0.1] * 3,
            "nll": 0.1,
        }
    ]
    result = stopped_result(task, forecast, 0.95)
    assert result["queries"] == 10 and result["found"] == 10
    assert result["early_loss"] == 0


def test_mean_ablation_uses_fractional_expected_discovery_allowance():
    from methods.baselines.lrt.evaluation.adaptive_report import mean_stopping_result

    task = {"budget": 20, "positives": 3, "found": [min(t, 3) for t in range(1, 21)]}
    forecast = [{"query": 10, "found": 3, "mean": [0.1, 0.2, 0.3]}]
    result = mean_stopping_result(task, forecast)
    assert result["queries"] == 10 and result["found"] == 3
