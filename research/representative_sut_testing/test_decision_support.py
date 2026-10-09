"""Joint decision uncertainty differs from independent point uncertainty."""
from types import SimpleNamespace
import json

import numpy as np
from scipy.special import ndtr
import torch

from .decision_support import bivariate_normal_probability, discovery_loss_support


def build(correlation):
    model = SimpleNamespace(mean=torch.tensor([-.5, .5], dtype=torch.float64),
                            covariance=torch.tensor([[1., correlation], [correlation, 1.]], dtype=torch.float64),
                            noise=torch.ones(2, dtype=torch.float64))
    return SimpleNamespace(models=[model], selection_weights=lambda: np.asarray([1.]))








def test_bivariate_probability_matches_independence_and_zero_mean_quadrant():
    np.testing.assert_allclose(bivariate_normal_probability(.3, -.5, 0), ndtr(.3) * ndtr(-.5), atol=1e-12)
    for correlation in (-.95, -.4, .4, .95):
        exact = .25 + np.arcsin(correlation) / (2 * np.pi)
        np.testing.assert_allclose(bivariate_normal_probability(0, 0, correlation), exact, atol=1e-10)


def test_almost_equivalent_high_success_actions_need_no_strict_order_identification():
    candidate = build(0)
    candidate.models[0].mean = torch.tensor([-4.1, -4.], dtype=torch.float64)
    loss = discovery_loss_support(candidate, 0, 1)
    assert loss['supported']
    assert loss['loss_probability'] < .003
    np.testing.assert_allclose(loss['gain_probability'] - loss['loss_probability'],
                               ndtr(4.1 / np.sqrt(2)) - ndtr(4 / np.sqrt(2)), atol=1e-10)
    json.dumps(loss, allow_nan=False)
