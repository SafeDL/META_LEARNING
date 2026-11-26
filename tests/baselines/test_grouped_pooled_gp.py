"""Grouping must preserve replicated likelihoods, gradients and GP predictions."""

import numpy as np
import pytest
import torch
from scipy.optimize import root
from scipy.special import expit

from testing_methods.baselines.response_baselines.grouped_gp import (
    GroupedJointPosterior,
    binomial_laplace,
    group_responses,
    predict_grouped,
)
from testing_methods.baselines.response_baselines.pooled_gp import predict_pooled
from testing_methods.history_transfer.inference import JointPosterior
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_grouping_keeps_counts_successes_and_rss_scatter_without_changing_data():
    x, labels, r = (
        np.array([[0.1], [0.1], [0.7]]),
        np.array([True, False, False]),
        np.array([0.2, 0.4, -0.3]),
    )
    before = x.copy(), labels.copy(), r.copy()
    grouped = group_responses(x, labels, r)
    assert grouped.counts.tolist() == [2, 1]
    assert grouped.successes.tolist() == [1, 0]
    assert grouped.rss_mean == pytest.approx([0.3, -0.3])
    assert grouped.rss_squared_residuals == pytest.approx([0.02, 0])
    shuffled = group_responses(x[::-1], labels[::-1], r[::-1])
    assert np.allclose(shuffled.rss_mean, grouped.rss_mean)
    assert all(np.array_equal(a, b) for a, b in zip(before, (x, labels, r)))


def test_counted_laplace_matches_an_independent_expanded_observation_newton_solution():
    mean = np.array([0.2, -0.4, 0.1])
    covariance = np.array([[1.2, 0.3, 0.1], [0.3, 0.9, 0.2], [0.1, 0.2, 1.1]])
    indices = np.repeat(np.arange(3), [3, 5, 2])
    labels = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    lower = np.linalg.cholesky(covariance + 1e-5 * np.eye(3))
    projected = lower[indices]

    def gradient(w):
        return w + projected.T @ (expit(mean[indices] + projected @ w) - labels)

    def hessian(w):
        p = expit(mean[indices] + projected @ w)
        return np.eye(3) + projected.T @ ((p * (1 - p))[:, None] * projected)

    solution = root(gradient, np.zeros(3), jac=hessian, tol=1e-11)
    assert np.max(np.abs(gradient(solution.x))) < 1e-9
    latent = mean[indices] + projected @ solution.x
    expected = (
        0.5 * solution.x @ solution.x
        + np.sum(np.logaddexp(0, latent) - labels * latent)
        + 0.5 * np.linalg.slogdet(hessian(solution.x))[1]
    )
    state = binomial_laplace(torch.tensor(mean), torch.tensor(covariance), [1, 4, 0], [3, 5, 2])
    assert float(state.nll) == pytest.approx(expected, abs=1e-10)
    assert np.allclose(state.alpha.numpy(), np.linalg.solve(lower.T, solution.x), atol=1e-10)


def records(repeated):
    unique = np.array([[0.1, 0.2], [0.4, 0.7], [0.9, 0.3]])
    if repeated:
        x = np.repeat(unique, 4, axis=0)
        labels = np.array(
            [False, False, False, True, True, True, False, True, False, False, False, False]
        )
        r = np.array([0.4, 0.3, 0.6, 0.5, -0.2, -0.3, 0.1, -0.1, 0.8, 0.6, 0.7, 0.9])
    else:
        x, labels, r = unique, np.array([False, True, False]), np.array([0.5, -0.2, 0.7])
    return x, labels, r


@pytest.mark.parametrize("repeated", [False, True])
def test_joint_grouped_and_dense_posteriors_agree_with_only_jitter_level_difference(repeated):
    x, labels, r = records(repeated)
    model = SharedPrivateKernel(2, deep=False).double()
    grouped = group_responses(x, labels, r)
    dense = JointPosterior(model.point_mean(len(x)), model.kernel(x), labels, r)
    compact = GroupedJointPosterior(
        model.point_mean(len(grouped.x)), model.kernel(grouped.x), grouped
    )
    query = np.array([[0.15, 0.25], [0.5, 0.5], [0.85, 0.4]])
    with torch.no_grad():
        dense_mean, dense_cov = dense.predict(
            model.point_mean(len(query)), model.kernel(query, x), model.point_covariance(len(query))
        )
        mean, covariance = compact.predict(
            model.point_mean(len(query)),
            model.kernel(query, grouped.x),
            model.point_covariance(len(query)),
        )
    tolerance = 1e-5 if repeated else 1e-10
    assert float(compact.r_nll.detach()) == pytest.approx(float(dense.r_nll.detach()), abs=1e-9)
    assert np.allclose(mean.numpy(), dense_mean.numpy(), atol=tolerance)
    assert np.allclose(covariance.numpy(), dense_cov.numpy(), atol=tolerance)
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-9
    assert compact.info["original_records"] == len(x)


@pytest.mark.parametrize(
    "name",
    tuple(
        n for n, _ in SharedPrivateKernel(2).named_parameters() if not n.startswith("embedding.")
    ),
)
def test_grouped_evidence_gradients_match_finite_differences(name):
    x, labels, r = records(True)
    grouped = group_responses(x, labels, r)
    model = SharedPrivateKernel(2, deep=False).double()

    def loss():
        return GroupedJointPosterior(
            model.point_mean(len(grouped.x)), model.kernel(grouped.x), grouped
        ).nll / len(x)

    parameter = getattr(model, name)
    analytical = float(torch.autograd.grad(loss(), parameter)[0])
    value = float(parameter.detach())
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            parameter.fill_(value + sign * 1e-5)
            values.append(float(loss()))
    assert analytical == pytest.approx((values[0] - values[1]) / 2e-5, rel=2e-4, abs=2e-6)


def test_no_replicates_produce_the_same_ten_step_fit_as_the_dense_baseline():
    x, labels, r = records(False)
    base = SharedPrivateKernel(2, deep=False).double()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    dense, parameters, _ = predict_pooled(base, x, labels, r, x)
    compact, grouped_parameters, info = predict_grouped(base, x, labels, r, x)
    assert compact == pytest.approx(dense, abs=1e-10)
    assert grouped_parameters == pytest.approx(parameters, abs=1e-10)
    assert info["original_records"] == info["unique_coordinates"] == 3
