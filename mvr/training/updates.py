"""Loss functions used by staged posterior, SAC, and PPO optimizers."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from typing import TYPE_CHECKING
from ..context.pearl_context import PearlContextEncoder
from ..context.outcome_schema import outcome_elbo
from ..policy.universal_scene_policy import UniversalScenePolicy
from .replay import InnerReplay, OuterRolloutBuffer

if TYPE_CHECKING:
    from ..model import TransferableScenarioMiner


def posterior_elbo(logits: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, *, kl_weight: float = 1e-3) -> torch.Tensor:
    return outcome_elbo(logits, target, kl=PearlContextEncoder.kl_to_prior(mean, logvar), kl_weight=kl_weight)


def clipped_ppo_loss(new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor, value: torch.Tensor, return_target: torch.Tensor, entropy: torch.Tensor, *, clip_ratio: float = 0.2, value_weight: float = 0.5, entropy_weight: float = 0.01) -> torch.Tensor:
    ratio = (new_logprob - old_logprob).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage).mean()
    critic = F.mse_loss(value, return_target)
    return policy + value_weight * critic - entropy_weight * entropy.mean()


def update_outer_ppo(
    policy: UniversalScenePolicy,
    rollout: OuterRolloutBuffer,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 4,
    batch_size: int = 64,
    router_balance_weight: float = 0.01,
) -> float:
    """Update a scene policy exclusively from one finished on-policy rollout."""
    if epochs < 1:
        raise ValueError("PPO requires at least one update epoch")
    device = next(policy.parameters()).device
    losses = []
    for _ in range(epochs):
        for batch in rollout.minibatches(batch_size):
            logprob, entropy, value = policy.evaluate(
                batch["scene_embedding"].to(device), batch["candidate_embeddings"].to(device),
                batch["candidate_mask"].to(device), batch["latent"].to(device), batch["expert"].to(device),
                batch["candidate"].to(device), batch["continuous"].to(device), batch["option"].to(device),
            )
            loss = clipped_ppo_loss(
                logprob, batch["old_logprob"].to(device), batch["advantage"].to(device), value,
                batch["return"].to(device), entropy,
            )
            router_logits = policy.router(
                batch["scene_embedding"].to(device), batch["latent"].to(device)
            )
            loss = loss + float(router_balance_weight) * policy.router.load_balance_loss(router_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


def update_inner_sac(model: "TransferableScenarioMiner", replay: InnerReplay, optimizer: torch.optim.Optimizer, *, batch_size: int = 64) -> dict[str, float]:
    """Update the Inner SAC from risk-reward replay with reconstructed features."""
    rows = replay.sample(batch_size)
    device = model.device
    maps = torch.stack([
        model.encode_scene(row.map_tokens, row.interactions).global_embedding for row in rows
    ])
    states = torch.as_tensor(np.stack([row.state for row in rows]), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack([row.next_state for row in rows]), dtype=torch.float32, device=device)
    latent = torch.stack([row.latent for row in rows]).to(device)
    option = torch.stack([row.option_index for row in rows]).to(device).long().reshape(-1)
    config = torch.stack([row.config for row in rows]).to(device)
    action = torch.as_tensor(np.stack([row.action for row in rows]), dtype=torch.float32, device=device)
    reward = torch.as_tensor([row.reward for row in rows], dtype=torch.float32, device=device)
    done = torch.as_tensor([row.done for row in rows], dtype=torch.bool, device=device)
    features = model.inner_features(states, maps, latent, option, config)
    next_features = model.inner_features(next_states, maps, latent, option, config)
    losses = model.inner_sac.losses(features, action, reward, next_features, done)
    total = losses.actor + losses.critic + losses.alpha
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    model.inner_sac.soft_update()
    return {"inner_actor_loss": float(losses.actor.detach().cpu()), "inner_critic_loss": float(losses.critic.detach().cpu()), "inner_alpha_loss": float(losses.alpha.detach().cpu())}
