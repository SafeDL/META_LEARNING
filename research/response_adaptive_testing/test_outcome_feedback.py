"""Verify probit observation moments and feedback isolation."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
import torch

from .outcome_feedback import OutcomeTestingSession, probit_factors
from .test_method import example


@pytest.mark.parametrize("mean", [-2, 0, 2])
@pytest.mark.parametrize("collision", [False, True])
def test_binary_observation_moments_match_direct_integration(mean, collision):
    variance = 0.8
    sign = 1 if collision else -1
    probability = ndtr(sign * mean / np.sqrt(1 + variance))
    first = quad(lambda z: (mean + np.sqrt(variance) * z) * ndtr(sign * (
        mean + np.sqrt(variance) * z)) * norm.pdf(z),
                 -10,
                 10,
                 epsabs=1e-12)[0] / probability
    second = quad(lambda z: (mean + np.sqrt(variance) * z)**2 * ndtr(sign * (
        mean + np.sqrt(variance) * z)) * norm.pdf(z),
                  -10,
                  10,
                  epsabs=1e-12)[0] / probability
    mean_factor, covariance_factor = probit_factors(
        torch.tensor(float(mean), dtype=torch.float64),
        torch.tensor(variance, dtype=torch.float64), collision)
    assert mean + variance * float(mean_factor) == pytest.approx(first,
                                                                 abs=1e-11)
    assert variance - variance**2 * float(covariance_factor) == pytest.approx(
        second - first**2, abs=1e-11)


def session(use_risk):
    x, risks, collisions, options = example()
    options.update({"transform": "logit", "collision_transform": "probit"})
    return OutcomeTestingSession(x, (risks, collisions),
                                 options,
                                 use_risk=use_risk,
                                 budget=2,
                                 device="cpu")


def test_combined_observations_contract_joint_covariance_without_losing_psd():
    model = session(True)
    before = torch.cat((torch.cat(
        (model.rr, model.cr.T), 1), torch.cat((model.cr, model.cc), 1)), 0)
    for risk, collision in ((0.8, True), (0.2, False)):
        model.next_index()
        model.observe(risk, collision)
    after = torch.cat((torch.cat(
        (model.rr, model.cr.T), 1), torch.cat((model.cr, model.cc), 1)), 0)
    assert torch.linalg.eigvalsh(after).min() > -1e-12
    assert torch.linalg.eigvalsh(before - after).min() > -1e-12
    assert torch.all(after.diag() <= before.diag() + 1e-12)
    assert torch.isfinite(model.probabilities()).all()
    assert len({row["index"] for row in model.records}) == 2


def test_binary_only_policy_is_independent_of_observed_risk_values():
    left, right = session(False), session(False)
    for label in (True, False):
        assert left.next_index() == right.next_index()
        left.observe(0.1, label)
        right.observe(0.9, label)
        torch.testing.assert_close(left.probabilities(),
                                   right.probabilities(),
                                   atol=0,
                                   rtol=0)
