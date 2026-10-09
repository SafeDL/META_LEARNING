"""Verify exact hypothesis likelihoods and cross-family behavioral inference."""
import numpy as np
import torch

from .session import BehaviorTestingSession, rmse_discrepancy


def test_sequential_hypothesis_weights_equal_batch_gaussian_likelihood():
    x = np.array([[0., 0., 0., 0., 0.], [0.1, 0.2, 0., 0., 0.],
                  [0., 0., 0., 0., 1.]])
    risk = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.4], [0.7, 0.8, 0.4]])
    collision = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.75], [0.7, 0.8, 0.9]])
    session = BehaviorTestingSession(x,
                                     risk,
                                     collision,
                                     rmse_discrepancy([0.01, 0.02]),
                                     budget=2,
                                     device="cpu")
    prior = session.rr.clone()
    noise = session.noise.clone()
    for index, value in ((0, 0.45), (1, 0.55)):
        session.pending = index
        session.records.append({})
        session.observe(value)
    covariance = prior[:2, :2] + torch.diag(noise[:2])
    residual = torch.tensor([0.45, 0.55],
                            dtype=torch.float64)[None, :] - torch.tensor(
                                risk[:, :2], dtype=torch.float64)
    likelihood = -0.5 * (residual *
                         torch.linalg.solve(covariance, residual.T).T).sum(1)
    expected = likelihood.softmax(0)
    torch.testing.assert_close(session.log_weights.exp(),
                               expected,
                               atol=1e-12,
                               rtol=0)
    mean = torch.tensor(risk,
                        dtype=torch.float64) + residual @ torch.linalg.solve(
                            covariance, prior[:2])
    torch.testing.assert_close(session.risk_mean, mean, atol=1e-12, rtol=0)
    expected_q = expected @ torch.tensor(collision, dtype=torch.float64)
    torch.testing.assert_close(session.probabilities(),
                               expected_q,
                               atol=1e-12,
                               rtol=0)
    assert abs(float(expected_q[2]) - float(collision[:, 2].mean())) > 0.01
    assert torch.linalg.eigvalsh(session.rr).min() > -1e-12


def test_identical_collision_predictions_are_unchanged_by_risk_identification(
):
    x = np.array([[0., 0., 0., 0., 0.], [0.1, 0.2, 0., 0., 1.]])
    risk = np.array([[0.1, 0.2], [0.7, 0.8]])
    collision = np.array([[0.4, 0.6], [0.4, 0.6]])
    session = BehaviorTestingSession(x,
                                     risk,
                                     collision,
                                     rmse_discrepancy([0.01, 0.01]),
                                     budget=1,
                                     device="cpu")
    before = session.probabilities().clone()
    index = session.next_index()
    session.observe(0.15)
    torch.testing.assert_close(session.probabilities(),
                               before,
                               atol=1e-12,
                               rtol=0)
    assert float(session.log_weights.exp().max()) > 0.99
    assert session.next_index() is None
