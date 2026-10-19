"""Student-t lookahead accuracy and the information value of correlation."""

import numpy as np
import pytest
import torch

from methods.baselines.cg_brt.posterior import RiskPosterior
from methods.baselines.dart.fantasy import fantasy_probabilities
from methods.baselines.dart.planning import top_sum


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA quadrature requires CUDA"
)
def test_accelerated_continuation_agrees_with_exact_student_cdf():
    from methods.baselines.dart.acceleration import continuation_values

    x = np.linspace(0, 1, 60)
    covariance = 0.04 * np.exp(-(((x[:, None] - x[None, :]) / 0.2) ** 2))
    priors = np.column_stack((0.3 + 0.4 * x, 0.7 - 0.4 * x))
    model = RiskPosterior(covariance, priors, [0.5, 0.5], priors.mean(1))
    model.observe(0, 0.4)
    candidates = np.array([2, 9, 35])
    available = np.arange(1, 60)
    draws = model.draws(candidates, 16, np.random.default_rng(11), 1.0)
    for spatial in (True, False):
        gpu = continuation_values(model, candidates, draws, available, 10, spatial)
        exact = []
        for column, candidate in enumerate(candidates):
            future = available[available != candidate]
            p = fantasy_probabilities(
                model, candidate, draws[:, column], future, spatial
            )
            exact.append(top_sum(p, 10).mean())
        np.testing.assert_allclose(gpu, exact, atol=1e-12)


def test_spatial_feedback_changes_information_value_with_identical_marginals():
    x = np.linspace(0, 1, 8)
    covariance = 0.04 * np.exp(-(((x[:, None] - x[None, :]) / 0.3) ** 2))
    priors = (0.47 + 0.06 * x)[:, None]
    model = RiskPosterior(covariance, priors, [1.0], priors[:, 0])
    candidates = np.array([1, 4])
    draws = model.draws(candidates, 1024, np.random.default_rng(29), 1.0)
    available = np.arange(8)
    spatial_values = []
    independent_values = []
    for i, a in enumerate(candidates):
        future = available[available != a]
        for spatial, values in ((True, spatial_values), (False, independent_values)):
            p = fantasy_probabilities(model, a, draws[:, i], future, spatial)
            values.append(top_sum(p, 3).mean())
    assert np.max(np.abs(np.array(spatial_values) - independent_values)) > 0.05
