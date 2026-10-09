import numpy as np
import pytest
import torch
from scipy.special import expit

from .session import RiskTestingSession


def test_shared_conditioning_matches_batch_gp():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 4))
    covariance = .02 * features @ features.T + .01 * np.eye(12)
    risks = rng.uniform(.1, .9, size=(12, 3))
    collisions = risks**2
    weights = np.array([.25, .25, .5])
    x = np.zeros((12, 5))
    calibration = {0: {"slope": 3., "intercept": -1.}}
    indices, observations = [3, 8, 2], [.2, .8, .5]
    prior = np.column_stack((risks @ weights, collisions @ weights))
    transformed = np.column_stack(
        (observations, expit(3 * np.asarray(observations) - 1)))
    block = covariance[np.ix_(indices, indices)] + .05 * np.eye(3)
    cross = covariance[:, indices]
    expected_mean = prior + cross @ np.linalg.solve(
        block, transformed - prior[indices])
    expected_covariance = covariance - cross @ np.linalg.solve(block, cross.T)
    for order in (range(3), reversed(range(3))):
        session = RiskTestingSession(x,
                                     covariance,
                                     risks,
                                     collisions,
                                     weights,
                                     calibration,
                                     device="cpu")
        for position in order:
            session.pending = indices[position]
            session.records.append({})
            session.observe(observations[position])
        np.testing.assert_allclose(session.mean.numpy(),
                                   expected_mean,
                                   atol=1e-12)
        np.testing.assert_allclose(session.covariance.numpy(),
                                   expected_covariance,
                                   atol=1e-12)
        assert session.count == 3


def test_budget_pending_and_missing_risk_contract():
    x = np.zeros((4, 5))
    risk = np.array([[.2], [.8], [.5], [.3]])
    session = RiskTestingSession(x,
                                 np.eye(4) * .01,
                                 risk,
                                 risk,
                                 np.ones(1),
                                 {0: {
                                     "slope": 3.,
                                     "intercept": -1.
                                 }},
                                 budget=2,
                                 device="cpu")
    with pytest.raises(RuntimeError):
        session.observe(.5)
    session.next_index()
    with pytest.raises(RuntimeError):
        session.next_index()
    before = session.mean.clone()
    session.observe(None)
    assert torch.equal(before, session.mean)
    assert session.count == 1
    session.next_index()
    session.observe(.4)
    assert session.count == 2 and session.next_index() is None
    assert session.records[0]["continuous_risk"] is None
    assert "collision" not in session.records[1]


def test_exact_sign_flip_small_sample():
    from .evaluate import paired
    assert paired([1.] * 5)["exact_two_sided_sign_flip_p"] == 2 / 32
    assert paired([0.] * 5)["exact_two_sided_sign_flip_p"] == 1.
