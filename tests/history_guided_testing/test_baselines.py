"""Six-baseline checks prepared for the next authorized test run."""
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from methods.history_guided_testing import baseline
from methods.history_guided_testing.scenarios import parameter_cells


def test_expected_improvement_matches_integrated_gaussian():
    mean = np.array([-0.3, 0.5, 1.2])
    variance = np.array([0.04, 0.25, 0.01])
    bound = np.array([0.5, 0.5, 0.9])
    reference = [quad(lambda y: (y - b) * norm.pdf(y, loc=m, scale=np.sqrt(v)), b, np.inf)[0]
                 for m, v, b in zip(mean, variance, bound)]
    np.testing.assert_allclose(baseline.expected_improvement(mean, variance, bound), reference, atol=1e-10)


def test_raw_risk_gp_matches_joint_conditioning():
    x = np.random.default_rng(17).random((8, 4))
    model = baseline.GaussianRiskGP(x)
    prior = model.covariance.copy()
    indices, risks = [2, 5, 0], np.array([0.1, 0.9, 0.4])
    for i, risk in zip(indices, risks):
        model.observe(i, risk)
    noise = baseline.SETTINGS["noise_variance"] + baseline.SETTINGS["jitter"]
    matrix = prior[np.ix_(indices, indices)] + noise * np.eye(len(indices))
    cross = prior[:, indices]
    expected_mean = baseline.SETTINGS["mean"] + cross @ np.linalg.solve(matrix, risks - baseline.SETTINGS["mean"])
    expected_covariance = prior - cross @ np.linalg.solve(matrix, cross.T)
    np.testing.assert_allclose(model.mean, expected_mean, atol=1e-10)
    np.testing.assert_allclose(model.covariance, expected_covariance, atol=1e-10)


@dataclass
class Observation:
    scenario_id: str
    risk: float
    collision: int
    valid_risk: bool = True


@pytest.mark.parametrize("method", tuple(baseline.METHODS))
def test_all_selectors_ignore_collision_labels_and_charge_initialization(monkeypatch, method):
    monkeypatch.setattr(baseline, "BUDGET", 6)
    monkeypatch.setitem(baseline.SETTINGS, "initial_queries", 2)
    x = np.random.default_rng(7).random((10, 5))
    x[:, 4] = np.arange(10) >= 5
    cells = parameter_cells(x)
    risks = np.linspace(0.1, 0.9, len(x))

    class Oracle:
        def __init__(self, label):
            self.label, self.queries = label, []

        def query(self, i):
            assert i not in self.queries
            self.queries.append(i)
            return Observation(str(i), float(risks[i]), self.label)

    safe, collision = Oracle(0), Oracle(1)
    scores = np.arange(len(x), dtype=float)
    first = baseline.select(method, x, cells, safe, 11, history_scores=scores)
    second = baseline.select(method, x, cells, collision, 11, history_scores=scores)
    assert first["selected_indices"] == second["selected_indices"]
    assert len(safe.queries) == len(set(safe.queries)) == 6
    assert len(first["queries"]) == 6
