"""Loss functions used by staged posterior, SAC, and PPO optimizers."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from ..context.set_posterior import SetPosterior
from ..context.outcome_schema import outcome_elbo
from ..policy.scene_policy import HybridScenePolicy
from .replay import OuterRolloutBuffer


def posterior_elbo(logits: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, *, kl_weight: float = 1e-3) -> torch.Tensor:
    return outcome_elbo(logits, target, kl=SetPosterior.kl_to_unit(mean, logvar), kl_weight=kl_weight)


def clipped_ppo_loss(new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor, value: torch.Tensor, return_target: torch.Tensor, entropy: torch.Tensor, *, clip_ratio: float = 0.2, value_weight: float = 0.5, entropy_weight: float = 0.01) -> torch.Tensor:
    ratio = (new_logprob - old_logprob).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage).mean()
    critic = F.mse_loss(value, return_target)
    return policy + value_weight * critic - entropy_weight * entropy.mean()


def update_outer_ppo(policy: HybridScenePolicy, rollout: OuterRolloutBuffer, optimizer: torch.optim.Optimizer, *, epochs: int = 4, batch_size: int = 64) -> float:
    """Update a scene policy exclusively from one finished on-policy rollout."""
    if epochs < 1:
        raise ValueError("PPO requires at least one update epoch")
    device = next(policy.parameters()).device
    losses = []
    for _ in range(epochs):
        for batch in rollout.minibatches(batch_size):
            inputs = torch.cat((batch["map_embedding"], batch["latent"], batch["history"]), dim=-1).to(device)
            logprob, entropy, value = policy.evaluate(inputs, batch["candidate"].to(device), batch["continuous"].to(device), batch["option"].to(device))
            loss = clipped_ppo_loss(logprob, batch["old_logprob"].to(device), batch["advantage"].to(device), value, batch["return"].to(device), entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0
