"""A captured numerical failure must converge without relaxing the score criterion."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import root
from scipy.special import expit

from testing_methods.baselines.response_baselines import grouped_gp as module


@pytest.fixture(scope="module")
def conditional():
    torch.set_num_threads(1)
    path = Path(__file__).parents[1] / "fixtures/pooled_mode_roundoff.npz"
    with np.load(path) as data:
        return {k: torch.tensor(data[k], dtype=torch.float64) for k in data.files}


def test_captured_roundoff_failure_reaches_the_unchanged_score_tolerance(conditional):
    state = module.binomial_laplace(**conditional)
    m, c, n, y = (conditional[k].numpy() for k in ("mean", "covariance", "counts", "successes"))
    lower = np.linalg.cholesky(c + 1e-5 * np.eye(len(m)))
    w = lower.T @ state.alpha.detach().numpy()
    independent_score = w + lower.T @ (n * expit(m + lower @ w) - y)
    assert module.MODE_TOLERANCE == 1e-10
    assert state.gradient_max <= module.MODE_TOLERANCE
    assert abs(independent_score).max() <= 1e-10
    assert state.precision_steps >= 1


def test_refined_evidence_matches_an_independent_scipy_stationary_solution(conditional):
    state = module.binomial_laplace(**conditional)
    m, c, n, y = (conditional[k].numpy() for k in ("mean", "covariance", "counts", "successes"))
    lower = np.linalg.cholesky(c + 1e-5 * np.eye(len(m)))

    def gradient(w):
        return w + lower.T @ (n * expit(m + lower @ w) - y)

    def hessian(w):
        p = expit(m + lower @ w)
        return np.eye(len(m)) + lower.T @ ((n * p * (1 - p))[:, None] * lower)

    start = lower.T @ state.alpha.detach().numpy() + 1e-4 * np.sin(np.arange(len(m)))
    solution = root(gradient, start, jac=hessian, tol=1e-11)
    assert abs(gradient(solution.x)).max() < 1e-10
    latent = m + lower @ solution.x
    evidence = (
        0.5 * solution.x @ solution.x
        + (n * np.logaddexp(0, latent) - y * latent).sum()
        + 0.5 * np.linalg.slogdet(hessian(solution.x))[1]
    )
    assert float(state.nll.detach()) == pytest.approx(evidence, abs=1e-9)
    np.testing.assert_allclose(
        state.alpha.detach().numpy(), np.linalg.solve(lower.T, solution.x), rtol=0, atol=1e-9
    )


def test_precision_refinement_preserves_differentiable_evidence(conditional):
    theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def loss(value):
        return module.binomial_laplace(
            conditional["mean"] + value[0],
            conditional["covariance"] * value[1].exp(),
            conditional["successes"],
            conditional["counts"],
        ).nll

    actual = torch.autograd.grad(loss(theta), theta)[0].detach().numpy()
    numerical = []
    for index in range(2):
        delta = torch.zeros(2, dtype=torch.float64)
        delta[index] = 2e-5
        numerical.append(float((loss(delta) - loss(-delta)).detach()) / 4e-5)
    np.testing.assert_allclose(actual, numerical, rtol=5e-4, atol=1e-5)


def test_exhausted_iterations_do_not_turn_into_false_convergence(conditional, monkeypatch):
    monkeypatch.setattr(module, "MODE_ITERATIONS", 1)
    with pytest.raises(FloatingPointError, match="did not converge"):
        module.binomial_laplace(**conditional)
