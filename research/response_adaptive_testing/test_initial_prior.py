"""Check the mean-only intervention and whole-source exclusion."""
import torch

from .initial_prior import reference_means


def test_uniform_reference_recovers_empirical_means():
    generator = torch.Generator().manual_seed(11)
    family = torch.tensor([0, 1, 0, 1, 0, 1])
    risk = torch.randn((6, 4), dtype=torch.float64, generator=generator)
    collision = torch.randn((6, 4), dtype=torch.float64, generator=generator)
    logits = torch.zeros((2, 2, 6), dtype=torch.float64)
    offsets = torch.tensor([-0.1, -0.2], dtype=torch.float64)
    mr, mc = reference_means(family, risk, collision, [0, 2, 3, 5], logits,
                             offsets)
    torch.testing.assert_close(mr, risk.mean(1), atol=1e-14, rtol=0)
    torch.testing.assert_close(mc,
                               collision.mean(1) + offsets[family],
                               atol=1e-14,
                               rtol=0)


def test_excluded_source_logits_cannot_change_means_or_gradients():
    generator = torch.Generator().manual_seed(23)
    family = torch.tensor([0, 1, 0, 1, 0, 1])
    risk = torch.randn((6, 3), dtype=torch.float64, generator=generator)
    collision = torch.randn((6, 3), dtype=torch.float64, generator=generator)
    logits = torch.randn((2, 2, 6),
                         dtype=torch.float64,
                         generator=generator,
                         requires_grad=True)
    offsets = torch.zeros(2, dtype=torch.float64)
    means = reference_means(family, risk, collision, [0, 2, 4], logits,
                            offsets)
    changed = logits.detach().clone()
    changed[:, :, [1, 3, 5]] = 100
    changed_means = reference_means(family, risk, collision, [0, 2, 4],
                                    changed, offsets)
    for expected, actual in zip(means, changed_means):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    sum(mean.square().sum() for mean in means).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, :, [1, 3, 5]]) == 0
