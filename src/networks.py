"""
Network architectures for the adversarial NPC meta-learning project.

Architecture contract (from FINAL_PROPOSAL.md):
  g_phi  : shared encoder  [38 → 256 → 256]  (tanh activations)
  Actor  : g_phi output  → [256 → 256 → act_dim]  (per-algorithm head)
  Critics: fully independent MLPs on raw (obs, action)  [obs+act → 256 → 256 → 1]

Transfer rule:
  - Load g_phi weights into actor encoder trunk.
  - All critic/value parameters are freshly Xavier-initialised per algorithm.
  - This applies to PPO (value head), SAC (two Q-nets), and TD3 (two Q-nets).
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

OBS_DIM = 38
ACT_DIM = 2
HIDDEN   = 256


# ─── Utility ─────────────────────────────────────────────────────────────────

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def xavier_init(module):
    """Apply Xavier uniform init to all Linear layers in *module*."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return module


# ─── Shared encoder g_phi ────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    g_phi: [OBS_DIM → HIDDEN → HIDDEN]  (tanh activations)

    This is the only module whose weights are transferred between meta-training
    and adaptation.  All other modules are freshly initialised per run.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], activation=nn.Tanh)
        self.out_dim = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─── PPO networks ────────────────────────────────────────────────────────────

class PPOActor(nn.Module):
    """
    Gaussian actor:  g_phi(obs) → head → mean + log_std
    log_std is a learned parameter (not input-dependent).
    """

    def __init__(self, encoder: SharedEncoder, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.encoder = encoder
        self.head    = mlp([hidden, hidden, act_dim], activation=nn.Tanh,
                           output_activation=nn.Identity)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        xavier_init(self.head)

    def forward(self, obs: torch.Tensor):
        z    = self.encoder(obs)
        mean = self.head(z)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor):
        dist   = self(obs)
        action = dist.rsample()
        log_p  = dist.log_prob(action).sum(-1)
        return action, log_p, dist.entropy().sum(-1)


class PPOCritic(nn.Module):
    """
    Value function: raw obs → [256 → 256 → 1].
    Fully independent of g_phi (freshly Xavier-initialised).
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1], activation=nn.Tanh)
        xavier_init(self.net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PPOActorCritic(nn.Module):
    """Convenience wrapper used during meta-training and PPO adaptation."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.encoder = SharedEncoder(obs_dim, hidden)
        self.actor   = PPOActor(self.encoder, act_dim, hidden)
        self.critic  = PPOCritic(obs_dim, hidden)

    def get_action(self, obs):
        return self.actor.get_action(obs)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        dist    = self.actor(obs)
        if action is None:
            action = dist.rsample()
        log_p   = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value   = self.critic(obs)
        return action, log_p, entropy, value


# ─── SAC networks ────────────────────────────────────────────────────────────

LOG_STD_MIN = -5
LOG_STD_MAX =  2


class SACActorNetwork(nn.Module):
    """
    Squashed Gaussian actor:  g_phi(obs) → head → (mean, log_std)
    Output clipped to [-1, 1] via tanh.
    """

    def __init__(self, encoder: SharedEncoder, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.encoder = encoder
        self.mean_head    = mlp([hidden, hidden, act_dim],
                                activation=nn.ReLU,
                                output_activation=nn.Identity)
        self.log_std_head = mlp([hidden, hidden, act_dim],
                                activation=nn.ReLU,
                                output_activation=nn.Identity)
        xavier_init(self.mean_head)
        xavier_init(self.log_std_head)

    def forward(self, obs: torch.Tensor):
        z       = self.encoder(obs)
        mean    = self.mean_head(z)
        log_std = self.log_std_head(z).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor):
        dist   = self(obs)
        x_t    = dist.rsample()
        y_t    = torch.tanh(x_t)
        # log prob with tanh correction
        log_p  = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_p  = log_p.sum(-1)
        return y_t, log_p, torch.tanh(dist.mean)


class SACQNetwork(nn.Module):
    """
    Q-function: raw (obs, action) → [256 → 256 → 1].
    Fully independent of g_phi.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU)
        xavier_init(self.net)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ─── TD3 networks ────────────────────────────────────────────────────────────

class TD3ActorNetwork(nn.Module):
    """
    Deterministic actor:  g_phi(obs) → head → action (tanh squashed)
    """

    def __init__(self, encoder: SharedEncoder, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.encoder = encoder
        self.head    = mlp([hidden, hidden, act_dim],
                           activation=nn.ReLU,
                           output_activation=nn.Tanh)
        xavier_init(self.head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs))


class TD3QNetwork(nn.Module):
    """
    Q-function: raw (obs, action) → [256 → 256 → 1].
    Fully independent of g_phi.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU)
        xavier_init(self.net)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ─── Weight transfer helpers ─────────────────────────────────────────────────

def load_encoder_weights(target_encoder: SharedEncoder, checkpoint_path: str,
                         device: str = "cpu") -> None:
    """
    Load only g_phi weights from a checkpoint into *target_encoder*.

    Checkpoint format produced by save_checkpoint():
      {"encoder": state_dict, ...}
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    enc_state = ckpt.get("encoder", ckpt)   # fallback: whole ckpt is encoder
    target_encoder.load_state_dict(enc_state, strict=True)


def save_checkpoint(encoder: SharedEncoder, path: str,
                    extra: Optional[dict] = None) -> None:
    """Save encoder weights (+ optional extras) to *path*."""
    payload = {"encoder": encoder.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
