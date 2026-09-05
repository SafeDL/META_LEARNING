"""Loss functions used by staged posterior, SAC, and PPO optimizers."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from typing import TYPE_CHECKING
from ..context.pearl_context import PearlContextEncoder
from ..context.outcome_schema import encode_outcome, outcome_elbo
from ..policy.universal_scene_policy import UniversalScenePolicy
from .replay import ContextReplay, InnerReplay, OuterRolloutBuffer, SupportGroup

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
                batch["candidate"].to(device), batch["continuous"].to(device),
                batch["continuous_mask"].to(device), batch["continuous_bounds"].to(device),
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


def _scene_contexts(
    model: "TransferableScenarioMiner", rows: list[object]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Rebuild current task and candidate encodings without replayed embeddings."""
    unique: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    keys = []
    for row in rows:
        domain = tuple(sorted(
            (str(name), tuple(float(value) for value in bounds))
            for name, bounds in row.logical_domain_bounds.items()
        ))
        key = str(row.geometry_hash), domain, tuple(bool(value) for value in row.logical_parameter_mask)
        keys.append(key)
        if key not in unique:
            encoded = model.encode_scene(row.map_tokens, row.interactions)
            unique[key] = (
                model.encode_task_structure(
                    encoded.global_embedding,
                    dict(row.logical_domain_bounds),
                    row.logical_parameter_mask,
                ),
                encoded.candidate_embeddings,
            )
    contexts = [unique[key] for key in keys]
    return torch.stack([value[0] for value in contexts]), [value[1] for value in contexts]


def _scene_embeddings(model: "TransferableScenarioMiner", rows: list[object]) -> torch.Tensor:
    return _scene_contexts(model, rows)[0]


def _concrete_inputs(
    model: "TransferableScenarioMiner", rows: list[object]
) -> tuple[torch.Tensor, torch.Tensor]:
    maps, candidate_embeddings = _scene_contexts(model, rows)
    candidates = torch.stack([
        embeddings[int(row.candidate_index)]
        for row, embeddings in zip(rows, candidate_embeddings)
    ])
    continuous = torch.as_tensor(
        np.asarray([row.continuous for row in rows]), dtype=torch.float32, device=model.device
    )
    masks = torch.as_tensor(
        np.asarray([row.logical_parameter_mask for row in rows]), dtype=torch.float32, device=model.device
    )
    return maps, model.concrete_features(candidates, continuous, masks)


def _episode_scene(model: "TransferableScenarioMiner", episode: object) -> torch.Tensor:
    encoded = model.encode_scene(episode.map_tokens, episode.interactions).global_embedding
    return model.encode_task_structure(
        encoded, dict(episode.logical_domain_bounds), episode.logical_parameter_mask
    )


def _episode_concrete(model: "TransferableScenarioMiner", episode: object) -> torch.Tensor:
    encoding = model.encode_scene(episode.map_tokens, episode.interactions)
    continuous = torch.as_tensor(episode.continuous, dtype=torch.float32, device=model.device)
    return model.concrete_features(
        encoding.candidate_embeddings[int(episode.candidate_index)],
        continuous,
        episode.logical_parameter_mask,
    ).squeeze(0)


def _group_latent(
    model: "TransferableScenarioMiner", group: SupportGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild one support posterior without copying tokens into transitions."""
    device = model.device
    tokens = []
    for episode in group.support_episodes:
        trajectory = episode.rollout.trajectory.to(device).unsqueeze(0)
        mask = torch.ones(trajectory.shape[:2], dtype=torch.bool, device=device)
        outcome = encode_outcome(episode.outcome).to(device).unsqueeze(0)
        tokens.append(model.episode_token_builder(
            _episode_scene(model, episode).unsqueeze(0),
            _episode_concrete(model, episode).unsqueeze(0), trajectory, mask, outcome,
        ).squeeze(0))
    support = torch.stack(tokens).unsqueeze(0)
    mean, logvar = model.infer_posterior(
        support, torch.ones((1, len(tokens)), dtype=torch.bool, device=device)
    )
    return mean.squeeze(0), logvar.squeeze(0)


def _posterior_group_loss(
    model: "TransferableScenarioMiner", group: SupportGroup, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    target = next(iter(group.query_episodes.values()))
    logits = model.outcome_decoder(
        mean.unsqueeze(0), _episode_scene(model, target).unsqueeze(0),
        _episode_concrete(model, target).unsqueeze(0),
    )
    return outcome_elbo(
        logits, encode_outcome(target.outcome).to(model.device).unsqueeze(0),
        kl=PearlContextEncoder.kl_to_prior(mean.unsqueeze(0), logvar.unsqueeze(0)),
        kl_weight=model.context_kl_weight,
    )


def update_inner_sac(
    model: "TransferableScenarioMiner",
    replay: InnerReplay,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 64,
    gradient_clip_norm: float = 5.0,
    event_sample_fraction: float = 0.25,
    event_action_weight: float = 0.0,
    gamma: float = 0.99,
    context_replay: ContextReplay | None = None,
) -> dict[str, float]:
    """Update the Inner SAC from risk-reward replay with reconstructed features."""
    rows = replay.sample(
        batch_size,
        positive_fraction=event_sample_fraction,
    )
    device = model.device
    states = torch.as_tensor(np.stack([row.state for row in rows]), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack([row.next_state for row in rows]), dtype=torch.float32, device=device)
    group_latents: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    posterior_losses: list[torch.Tensor] = []
    latent_rows = []
    for row in rows:
        if row.support_group_id is None:
            latent_rows.append(row.latent.to(device))
            continue
        if context_replay is None:
            raise ValueError("query transition requires context replay")
        group_id = row.support_group_id
        if group_id not in group_latents:
            group = context_replay.get(group_id)
            if group.task_id != row.task_id:
                raise ValueError("support group task does not match query transition")
            group_latents[group_id] = _group_latent(model, group)
            posterior_losses.append(_posterior_group_loss(
                model, group, *group_latents[group_id]
            ))
        latent_rows.append(group_latents[group_id][0])
    latent = torch.stack(latent_rows)
    action = torch.as_tensor(np.stack([row.action for row in rows]), dtype=torch.float32, device=device)
    reward = torch.as_tensor([row.reward for row in rows], dtype=torch.float32, device=device)
    done = torch.as_tensor([row.done for row in rows], dtype=torch.bool, device=device)
    duration_steps = torch.as_tensor(
        [row.duration_steps for row in rows], dtype=torch.float32, device=device
    )
    if bool((duration_steps < 1).any()):
        raise ValueError("inner transition duration_steps must be positive")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("inner SAC gamma must lie in (0, 1]")
    bootstrap_discount = torch.pow(
        torch.full_like(duration_steps, float(gamma)), duration_steps
    )
    maps, concrete = _concrete_inputs(model, rows)
    features = model.inner_features(states, maps, latent, concrete)
    next_features = model.inner_features(next_states, maps, latent, concrete)
    td_target = model.inner_sac.critic_target(
        reward, next_features, done, bootstrap_discount=bootstrap_discount,
        context=latent,
    )
    critic = model.inner_sac.critic_loss(
        features, action, reward, next_features, done,
        bootstrap_discount=bootstrap_discount, context=latent,
    )
    posterior = torch.stack(posterior_losses).mean() if posterior_losses else torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    (critic + posterior).backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        float(gradient_clip_norm),
    )
    optimizer.step()

    maps, concrete = _concrete_inputs(model, rows)
    features = model.inner_features(states, maps, latent.detach(), concrete)
    actor, alpha = model.inner_sac.actor_alpha_losses(
        features,
        actions=action,
        rewards=reward,
        event_action_weight=event_action_weight,
        context=latent.detach(),
    )
    optimizer.zero_grad(set_to_none=True)
    (actor + alpha).backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        float(gradient_clip_norm),
    )
    optimizer.step()
    model.inner_sac.soft_update()
    return {
        "inner_actor_loss": float(actor.detach().cpu()),
        "inner_critic_loss": float(critic.detach().cpu()),
        "inner_alpha_loss": float(alpha.detach().cpu()),
        "inner_td_target_variance": float(td_target.var(unbiased=False).detach().cpu()),
        "posterior_loss": float(posterior.detach().cpu()),
    }
