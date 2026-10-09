"""Verify the compact implementation against complete development runs."""
import numpy as np
import pytest
import torch

from methods.history_guided_testing.io import read_json
from research.history_response_testing.history_model import predict
from research.response_adaptive_testing.config import ROOT as DEVELOPMENT
from research.response_adaptive_testing.confirmation import CONFIRMATION

from .session import BUDGET, BudgetTestingSession


@pytest.mark.parametrize("name", ["idm_07", "fvdm_11"])
@pytest.mark.parametrize("seed", [11, 23, 37, 53, 71])
def test_compact_model_reproduces_all_two_hundred_development_queries(
        name, seed):
    torch.set_num_threads(1)
    state = read_json(DEVELOPMENT.parent / "budget_response_testing" /
                      "model_state.json")
    expected = read_json(DEVELOPMENT / "results" / "budget_training" /
                         "validation" / name / "pool_0" /
                         f"replay_budget_mean_200_{seed}.json")
    with np.load(CONFIRMATION / name / "pool_0" / "responses.npz") as bank:
        prediction = predict(bank["x"], seed)
        session = BudgetTestingSession(bank["x"], *prediction, state)
        selected = []
        while (index := session.next_index()) is not None:
            session.observe(float(bank["risk"][index]))
            selected.append(index)
        assert selected == expected["selected_indices"]
        assert len(selected) == len(set(selected)) == BUDGET
        assert [record["continuous_risk"] for record in session.records] == [
            record["continuous_risk"] for record in expected["observations"]
        ]


def test_pending_query_must_receive_one_risk_observation():
    state = read_json(DEVELOPMENT.parent / "budget_response_testing" /
                      "model_state.json")
    with np.load(CONFIRMATION / "idm_07" / "pool_0" / "responses.npz") as bank:
        prediction = predict(bank["x"][:8], 11)
        session = BudgetTestingSession(bank["x"][:8],
                                       *prediction,
                                       state,
                                       budget=1,
                                       device="cpu")
        session.next_index()
        with pytest.raises(RuntimeError, match="pending query"):
            session.next_index()
        session.observe(0.5)
        assert session.next_index() is None
        with pytest.raises(RuntimeError, match="before feedback"):
            session.observe(0.5)
