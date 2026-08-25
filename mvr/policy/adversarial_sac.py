"""Universal, option-conditioned SAC building blocks for the inner timescale."""
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

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normal = self.distribution(features)
        raw = normal.rsample()
        action = raw.tanh()
        log_prob = normal.log_prob(raw).sum(-1) - torch.log(1 - action.square() + 1e-6).sum(-1)
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


class OptionConditionedSAC(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int = 2, target_entropy: float | None = None) -> None:
        super().__init__()
        self.actor = _Actor(feature_dim, action_dim)
        self.critic1, self.critic2 = _Critic(feature_dim, action_dim), _Critic(feature_dim, action_dim)
        self.target1, self.target2 = _Critic(feature_dim, action_dim), _Critic(feature_dim, action_dim)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        distribution = self.actor.distribution(features)
        return (distribution.mean if deterministic else distribution.rsample()).tanh()

    def critic_loss(self, features: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_features: torch.Tensor, done: torch.Tensor, *, gamma: float = 0.99) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_logprob = self.actor.sample(next_features)
            next_q = torch.minimum(self.target1(next_features, next_action), self.target2(next_features, next_action)) - self.alpha.detach() * next_logprob
            target = reward + gamma * (1.0 - done.float()) * next_q
        return (self.critic1(features, action).sub(target).square() + self.critic2(features, action).sub(target).square()).mean()

    def actor_alpha_losses(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        critics = (*self.critic1.parameters(), *self.critic2.parameters())
        for parameter in critics:
            parameter.requires_grad_(False)
        try:
            sampled, logprob = self.actor.sample(features)
            actor = (self.alpha.detach() * logprob - torch.minimum(self.critic1(features, sampled), self.critic2(features, sampled))).mean()
        finally:
            for parameter in critics:
                parameter.requires_grad_(True)
        alpha = -(self.log_alpha * (logprob.detach() + self.target_entropy)).mean()
        return actor, alpha

    def losses(self, features: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_features: torch.Tensor, done: torch.Tensor, *, gamma: float = 0.99) -> SACLosses:
        critic = self.critic_loss(features, action, reward, next_features, done, gamma=gamma)
        actor, alpha = self.actor_alpha_losses(features)
        return SACLosses(actor, critic, alpha)

    @torch.no_grad()
    def soft_update(self, tau: float = 0.005) -> None:
        for target, source in ((self.target1, self.critic1), (self.target2, self.critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.lerp_(source_param, tau)
