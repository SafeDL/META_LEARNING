"""Check cached conditioning and the exact utility of a ranking swap."""
import numpy as np
import pytest
import torch

from .budget_training import (condition_parts, conditional_scores,
                              position_gains, ranking_loss, response_parts)
from .response_projection import projected_prior
from .test_method import example


@pytest.mark.parametrize("context", [[], [0], [0, 1]])
def test_cached_moments_equal_joint_conditioning_for_changed_loading(context):
    x, risks, collisions, options = example()
    options.update({
        "transform": "logit",
        "collision_transform": "probit",
        "residual_mode": "learned_sensitivity",
        "positive_loading": True,
        "loading": [[0.1, 0.2, -0.3, 0.1, 0.2, 0.5]] * 2,
        "collision_offset": [-0.1, -0.2],
        "sensitivity_regularization": 0.4
    })
    parts = response_parts(x, risks, collisions, options, "cpu")
    observed = np.array([0.6, 0.8])[:len(context)]
    episode = {**parts, **condition_parts(parts, context, observed, options)}
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn((2, 3), dtype=torch.float64, generator=generator)
    loading = torch.randn((2, 6), dtype=torch.float64, generator=generator)
    changed = {**options, "loading": loading}
    identity = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    mr, mc, rr, cr, cc, _ = projected_prior(x, risks, collisions, changed,
                                            identity, logits, [0, 1, 2])
    matrix = rr[context][:, context] + options["noise"] * torch.eye(
        len(context), dtype=rr.dtype)
    residual = mr.new_tensor(np.log(observed / (1 - observed))) - mr[context]
    cross = cr[:, context]
    mean = mc + cross @ torch.linalg.solve(matrix, residual)
    variance = cc - (cross * torch.linalg.solve(matrix, cross.T).T).sum(1)
    expected = mean / (1 + variance).sqrt()
    actual = conditional_scores(episode, logits, loading, options)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_position_gains_match_endpoint_and_discovery_area_swap():
    budget = 6
    count = 2
    tail = torch.tensor([0., 1., 0., 0., 1.])
    before = torch.cat((torch.tensor([1., 0.]), tail[:budget - count]))
    swapped = tail.clone()
    swapped[0], swapped[4] = tail[4], tail[0]
    after = torch.cat((torch.tensor([1., 0.]), swapped[:budget - count]))
    utility = lambda c: 0.5 * c.sum() + 0.5 * c.cumsum(0).mean()
    gains = position_gains(len(tail), budget - count, budget)
    torch.testing.assert_close(
        utility(after) - utility(before), (gains[0] - gains[4]).float())


@pytest.mark.parametrize("labels",
                         [[False] * 3, [True] * 3, [True, False, True]])
def test_budget_loss_has_finite_gradients_including_single_class(labels):
    score = torch.tensor([-12., 0., 12.],
                         dtype=torch.float64,
                         requires_grad=True)
    loss = ranking_loss(score, torch.tensor(labels), 2, weighted=True)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(score.grad).all()
