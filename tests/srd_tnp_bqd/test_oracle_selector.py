import copy

import numpy as np
import pytest
import torch

from methods.srd_tnp_bqd.common import DEFAULT_CONFIG, config_at
from methods.srd_tnp_bqd.data import HistoryOutput
from methods.srd_tnp_bqd.acquisition import Selector
from methods.srd_tnp_bqd.nonstationary_kernel import ResidualKernel
from methods.srd_tnp_bqd.oracle import ContinuousOracle


def test_oracle_billing_invalid_risk_and_duplicate():
    o = ContinuousOracle(["a", "b", "c"], lambda i: (int(i == 0), None, False), 2)
    with pytest.raises(ValueError):
        o.query(-1)
    a = o.query(0)
    assert a.collision == 1 and a.risk is None and a.query_number == 1
    with pytest.raises(ValueError):
        o.query(0)
    o.query(1)
    with pytest.raises(ValueError):
        o.query(2)


def test_unqueried_values_cannot_change_next_selection_and_exact_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(11)
    x = rng.random((205, 4))
    h = rng.normal(size=(205, 32))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    prior = HistoryOutput(np.zeros((205, 1)), h)
    torch.manual_seed(11)
    kernel = ResidualKernel(config_at(DEFAULT_CONFIG)["residual_gp"])
    seen = []
    def provider(i):
        seen.append(i)
        return 0, .4, True
    selector = Selector(prior, copy.deepcopy(kernel))
    oracle = ContinuousOracle([str(i) for i in range(205)], provider, 200)
    run = selector.run(x, np.zeros(205, int), oracle, 200)
    assert len(set(run["selected_indices"])) == len(oracle.queried) == 200
    assert run["risk_archive"]["risk_occupied_cells"] == 0
    first = run["selected_indices"][0]
    second = run["selected_indices"][1]
    # Only the first disclosed feedback remains identical; every unseen value
    # is changed, and the selector's next query must still be identical.
    changed = ContinuousOracle([str(i) for i in range(205)], lambda i: (0, .4 if i == first else .99, True), 2)
    other = Selector(prior, copy.deepcopy(kernel)).run(x, np.zeros(205, int), changed, 2)
    assert other["selected_indices"] == [first, second]
    torch.set_num_threads(previous)
