from __future__ import annotations

import torch

from meta_testing.context.set_posterior import SetPosterior
from meta_testing.context.trajectory_encoder import TrajectoryEncoder
from meta_testing.policy.adversarial_sac import OptionConditionedSAC
from meta_testing.policy.scene_policy import HybridScenePolicy


def test_trajectory_masks_and_set_posterior_permutation_invariance() -> None:
    encoder = TrajectoryEncoder(hidden_dim=16)
    sequence = torch.randn(2, 4, 12)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    assert encoder(sequence, mask).shape == (2, 16)
    posterior = SetPosterior(token_dim=8, latent_dim=4).eval()
    tokens = torch.randn(1, 3, 8)
    token_mask = torch.tensor([[True, True, True]])
    mean, logvar = posterior(tokens, token_mask)
    swapped_mean, swapped_logvar = posterior(tokens[:, [2, 0, 1]], token_mask)
    torch.testing.assert_close(mean, swapped_mean)
    torch.testing.assert_close(logvar, swapped_logvar)
    prior_mean, prior_logvar = posterior(tokens, torch.zeros_like(token_mask))
    torch.testing.assert_close(prior_mean, torch.zeros_like(prior_mean))
    torch.testing.assert_close(prior_logvar, torch.zeros_like(prior_logvar))


def test_inner_sac_and_outer_hybrid_policy_shapes() -> None:
    sac = OptionConditionedSAC(10, 2)
    features, next_features = torch.randn(4, 10), torch.randn(4, 10)
    losses = sac.losses(features, torch.tanh(torch.randn(4, 2)), torch.randn(4), next_features, torch.zeros(4))
    (losses.actor + losses.critic + losses.alpha).backward()
    outer = HybridScenePolicy(12, 3, 4, 4)
    action = outer.sample(torch.randn(2, 12))
    assert action.continuous.shape == (2, 4)
    assert outer.evaluate(torch.randn(2, 12), action.candidate_index, action.continuous, action.option_index)[0].shape == (2,)
