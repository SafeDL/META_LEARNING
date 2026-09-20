from __future__ import annotations

import numpy as np
import pytest
import torch

from replications.fst_highway_env.fst.fluctuation_audit import signed_fluctuation
from replications.fst_highway_env.fst.fusion import estimate, weights_from_attention
from replications.fst_highway_env.fst.reference_distribution import (
    uniform_distribution,
    validate_distribution,
)
from replications.fst_highway_env.fst.set_optimizer import discrete_single_swap
from replications.fst_highway_env.fst.similarity_network import (
    SimilarityNetwork,
    minimax_loss_per_set,
    source_estimates,
)


def _fixture() -> tuple[SimilarityNetwork, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    features = torch.randn(12, 4)
    probability = torch.full((12,), 1 / 12)
    response = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    model = SimilarityNetwork(4, hidden_dim=16, linear_layers=4, distance_epsilon=1e-3)
    return model, features, probability, response


def test_query_normalization_and_probability_mass_conservation() -> None:
    model, features, probability, _response = _fixture()
    _embedding, attention, weights = model(features, torch.tensor([[1, 4, 8]]), probability)
    assert attention.shape == (1, 3, 12)
    assert torch.allclose(attention.sum(dim=1), torch.ones((1, 12)), atol=1e-6)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-6)


def test_query_permutation_permutes_weights_but_not_estimate() -> None:
    model, features, probability, response = _fixture()
    first = torch.tensor([[1, 4, 8]])
    second = torch.tensor([[8, 1, 4]])
    _, _, weights_first = model(features, first, probability)
    _, _, weights_second = model(features, second, probability)
    assert torch.allclose(weights_second, weights_first[:, [2, 0, 1]], atol=1e-6)
    estimate_first = source_estimates(weights_first, response, first)
    estimate_second = source_estimates(weights_second, response, second)
    assert torch.allclose(estimate_first, estimate_second, atol=1e-6)


def test_constant_response_and_n_one_are_exact() -> None:
    model, features, probability, _response = _fixture()
    selected = torch.tensor([[5]])
    _, attention, weights = model(features, selected, probability)
    assert torch.allclose(attention, torch.ones_like(attention))
    assert torch.allclose(weights, torch.ones_like(weights))
    constant = torch.full((2, 12), 0.37)
    estimates = source_estimates(weights, constant, selected)
    assert torch.allclose(estimates, torch.full((1, 2), 0.37))


def test_inverse_distance_backward_is_finite_and_nonzero() -> None:
    model, features, probability, response = _fixture()
    selected = torch.tensor([[1, 4, 8], [0, 6, 11]])
    _, _attention, weights = model(features, selected, probability)
    loss, _ = minimax_loss_per_set(weights, response, selected, probability)
    loss.mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.all(torch.isfinite(value)) for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0.0


def test_convex_hull_error_bound() -> None:
    source = np.asarray(
        [[0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1]],
        dtype=float,
    )
    p = uniform_distribution(source.shape[1])
    selected = np.asarray([0, 2, 5])
    weights = np.asarray([0.2, 0.35, 0.45])
    source_error = np.abs(estimate(source, selected, weights) - source @ p)
    coefficients = np.asarray([0.15, 0.25, 0.60])
    target = coefficients @ source
    target_error = abs(float(estimate(target, selected, weights)[0] - target @ p))
    assert target_error <= source_error.max() + 1e-12


def test_arbitrary_zero_response_set_cannot_recover_positive_mu() -> None:
    response = np.asarray([0, 0, 0, 1, 1], dtype=float)
    p = uniform_distribution(len(response))
    selected = np.asarray([0, 1, 2])
    weights = np.asarray([0.2, 0.3, 0.5])
    assert estimate(response, selected, weights)[0] == 0.0
    assert response @ p > 0.0


def test_signed_fluctuation_identity() -> None:
    response = np.asarray([0.0, 0.2, 0.7, 1.0, 0.4])
    selected = np.asarray([1, 3])
    attention = np.asarray([[0.8, 0.7, 0.2, 0.1, 0.4], [0.2, 0.3, 0.8, 0.9, 0.6]])
    p = uniform_distribution(len(response))
    _fluctuation, weights, residual = signed_fluctuation(
        response, selected, attention, p
    )
    assert np.isclose(weights.sum(), 1.0)
    assert abs(residual) < 1e-12


def test_single_swap_recomputes_and_improves_whole_set() -> None:
    target = {2, 4}

    def loss(sets: np.ndarray) -> np.ndarray:
        return np.asarray([len(target.symmetric_difference(set(row.tolist()))) for row in sets], dtype=float)

    result = discrete_single_swap(np.asarray([0, 1]), 6, loss, max_rounds=3)
    assert set(result.indices.tolist()) == target
    assert result.loss == 0.0
    assert result.candidate_evaluations > 1


def test_distribution_rejects_invalid_mass() -> None:
    validate_distribution(np.asarray([0.4, 0.6]), 2)
    with pytest.raises(ValueError):
        validate_distribution(np.asarray([0.4, 0.5]), 2)
