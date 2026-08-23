from __future__ import annotations

import torch

from mvr.context.outcome_decoder import PosteriorTrainingBatch
from mvr.context.pearl_context import PearlContextEncoder


def test_pearl_context_has_exact_prior_permutation_invariance_and_sampling() -> None:
    encoder = PearlContextEncoder(8, 4)
    tokens = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, True]])
    mean, logvar = encoder(tokens, mask)
    swapped = encoder(tokens[:, [2, 0, 1]], mask)
    torch.testing.assert_close(mean, swapped[0])
    torch.testing.assert_close(logvar, swapped[1])
    prior = encoder(tokens, torch.zeros_like(mask))
    torch.testing.assert_close(prior[0], torch.zeros_like(prior[0]))
    torch.testing.assert_close(prior[1], torch.zeros_like(prior[1]))
    assert not torch.allclose(encoder.sample(mean, logvar), mean)
    assert torch.isfinite(encoder.kl_to_prior(mean, logvar)).all()


def test_posterior_batch_rejects_support_target_overlap() -> None:
    batch = PosteriorTrainingBatch(
        torch.randn(1, 1, 8), torch.ones(1, 1, dtype=torch.bool), torch.randn(1, 4),
        torch.randn(1, 4), torch.zeros(1, dtype=torch.long), torch.zeros(1, 5),
        (("support",),), ("target",),
    )
    batch.validate()
    leaking = PosteriorTrainingBatch(
        batch.support_tokens, batch.support_mask, batch.target_scene, batch.target_config,
        batch.target_option, batch.target_outcome, (("target",),), ("target",),
    )
    try:
        leaking.validate()
    except ValueError as error:
        assert "must not appear" in str(error)
    else:
        raise AssertionError("support-target overlap must fail")
