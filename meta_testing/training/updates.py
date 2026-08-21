"""Loss functions used by staged posterior, SAC, and PPO optimizers."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from ..context.set_posterior import SetPosterior
from ..context.outcome_schema import outcome_elbo


def posterior_elbo(logits: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, *, kl_weight: float = 1e-3) -> torch.Tensor:
    return outcome_elbo(logits, target, kl=SetPosterior.kl_to_unit(mean, logvar), kl_weight=kl_weight)


def clipped_ppo_loss(new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor, value: torch.Tensor, return_target: torch.Tensor, entropy: torch.Tensor, *, clip_ratio: float = 0.2, value_weight: float = 0.5, entropy_weight: float = 0.01) -> torch.Tensor:
    ratio = (new_logprob - old_logprob).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage).mean()
    critic = F.mse_loss(value, return_target)
    return policy + value_weight * critic - entropy_weight * entropy.mean()
