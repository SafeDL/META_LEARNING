"""Continuous SAC building blocks for the Inner controller."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Normal


class _Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean, self.log_std = nn.Linear(256, action_dim), nn.Linear(256, action_dim)

    def distribution(self, features: torch.Tensor) -> Normal:
        body = self.body(features)
        return Normal(self.mean(body), self.log_std(body).clamp(-5.0, 2.0).exp())

    @classmethod
    def squash(cls, raw: torch.Tensor) -> torch.Tensor:
        return raw.tanh()

    def sample(
        self, features: torch.Tensor, raw_shift: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normal = self.distribution(features)
        base_raw = normal.rsample()
        raw = base_raw if raw_shift is None else base_raw + raw_shift
        action = self.squash(raw)
        log_prob = (
            normal.log_prob(base_raw).sum(-1)
            - torch.log(1 - action.square() + 1e-6).sum(-1)
        )
        return action, log_prob


class _Critic(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((features, action), dim=-1)).squeeze(-1)


@dataclass
class SACLosses:
    actor: torch.Tensor
    critic: torch.Tensor
    alpha: torch.Tensor


class AdversarialSAC(nn.Module):
    action_limit = 1.0

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 4,
        target_entropy: float | None = None,
        context_dim: int = 16,
    ) -> None:
        super().__init__()
        self.actor = _Actor(feature_dim, action_dim)
        self.critic1, self.critic2 = _Critic(feature_dim, action_dim), _Critic(feature_dim, action_dim)
        self.target1, self.target2 = _Critic(feature_dim, action_dim), _Critic(feature_dim, action_dim)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        if context_dim < 1:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.context_action = nn.Sequential(
            nn.Linear(self.context_dim, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, action_dim, bias=False),
        )
        # A zero/prior latent must preserve the shared prior exactly.  The
        # bias-free head then learns a bounded context-dependent residual in
        # the same pre-squash actuator space as the SAC actor.
        # Give the learned context residual enough pre-squash gain to make a
        # few-shot change observable; the final tanh and physical projector
        # remain the common action envelope for both prior and adapted
        # policies.
        self.context_action_scale = 64.0
        # The policy owns the full physical action. The shield, rather than a
        # hidden nominal controller, enforces its reachable action envelope.
        self.log_alpha = nn.Parameter(torch.tensor(-2.3025851))
        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _context_shift(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if context is None:
            return None
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[-1] != self.context_dim:
            raise ValueError("context latent dimension must match the SAC context head")
        return self.context_action_scale * torch.tanh(self.context_action(context))

    @torch.no_grad()
    def act(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        distribution = self.actor.distribution(features)
        raw = distribution.mean if deterministic else distribution.rsample()
        shift = self._context_shift(context)
        if shift is not None:
            raw = raw + shift
        return self.action_limit * self.actor.squash(raw)

    def critic_loss(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_features: torch.Tensor,
        done: torch.Tensor,
        *,
        gamma: float = 0.99,
        bootstrap_discount: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = self.critic_target(
            reward,
            next_features,
            done,
            gamma=gamma,
            bootstrap_discount=bootstrap_discount,
            context=context,
        )
        return (
            nn.functional.smooth_l1_loss(self.critic1(features, action), target)
            + nn.functional.smooth_l1_loss(self.critic2(features, action), target)
        )

    @torch.no_grad()
    def critic_target(
        self,
        reward: torch.Tensor,
        next_features: torch.Tensor,
        done: torch.Tensor,
        *,
        gamma: float = 0.99,
        bootstrap_discount: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the Bellman target used by both critics."""
        with torch.no_grad():
            next_action, next_logprob = self.actor.sample(
                next_features, self._context_shift(context)
            )
            next_action = self.action_limit * next_action
            next_q = torch.minimum(self.target1(next_features, next_action), self.target2(next_features, next_action)) - self.alpha.detach() * next_logprob
            discount = (
                torch.as_tensor(bootstrap_discount, dtype=reward.dtype, device=reward.device)
                if bootstrap_discount is not None
                else torch.as_tensor(gamma, dtype=reward.dtype, device=reward.device)
            )
            target = reward + discount * (1.0 - done.float()) * next_q
        return target

    def actor_alpha_losses(
        self,
        features: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
        event_action_weight: float = 0.0,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critics = (*self.critic1.parameters(), *self.critic2.parameters())
        for parameter in critics:
            parameter.requires_grad_(False)
        try:
            context_shift = self._context_shift(context)
            sampled, logprob = self.actor.sample(features, context_shift)
            sampled = self.action_limit * sampled
            q_value = torch.minimum(
                self.critic1(features, sampled), self.critic2(features, sampled)
            )
            actor = (self.alpha.detach() * logprob - q_value).mean()
            if event_action_weight > 0.0 and actions is not None and rewards is not None:
                # Shaping is capped below one; only captured valid events
                # include the shared terminal bonus above this threshold.
                event_mask = rewards >= 1.0
                if bool(event_mask.any()):
                    mean = self.actor.distribution(features).mean
                    if context_shift is not None:
                        mean = mean + context_shift
                    mean_action = self.action_limit * self.actor.squash(mean)
                    event_target = actions[event_mask].clamp(-self.action_limit, self.action_limit)
                    actor = actor + float(event_action_weight) * nn.functional.smooth_l1_loss(
                        mean_action[event_mask], event_target
                    )
        finally:
            for parameter in critics:
                parameter.requires_grad_(True)
        alpha = -(self.log_alpha * (logprob.detach() + self.target_entropy)).mean()
        return actor, alpha

    def losses(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_features: torch.Tensor,
        done: torch.Tensor,
        *,
        gamma: float = 0.99,
        bootstrap_discount: torch.Tensor | None = None,
    ) -> SACLosses:
        critic = self.critic_loss(
            features,
            action,
            reward,
            next_features,
            done,
            gamma=gamma,
            bootstrap_discount=bootstrap_discount,
        )
        actor, alpha = self.actor_alpha_losses(features)
        return SACLosses(actor, critic, alpha)

    @torch.no_grad()
    def soft_update(self, tau: float = 0.005) -> None:
        for target, source in ((self.target1, self.critic1), (self.target2, self.critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.lerp_(source_param, tau)
