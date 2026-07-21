"""PEARL-SAC implementation with prescribed gradient boundaries."""
from __future__ import annotations
import copy
import hashlib
from typing import Mapping
import numpy as np
import torch
from torch import nn

from .context_encoder import ContextEncoder
from .networks import Critic, GaussianActor
from .replay import Transition


class PEARLAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: Mapping[str, object], device: torch.device):
        self.device = device; pearl = config["pearl"]; networks = config["networks"]; sac = config["sac"]
        self.observation_schema = str(config["environment"]["observation_schema"]); self.action_dim = int(action_dim)
        self.latent_dim = int(pearl["latent_dim"]); self.reward_scale = float(pearl["context_reward_scale"]); self.gamma = float(sac["gamma"]); self.tau = float(sac["tau"]); self.kl_beta = float(pearl["kl_beta"])
        context_dim = observation_dim + action_dim + 1 + observation_dim + 2
        self.context_encoder = ContextEncoder(context_dim, self.latent_dim, list(networks["context_hidden_sizes"])).to(device)
        self.actor = GaussianActor(observation_dim, self.latent_dim, action_dim, list(networks["actor_hidden_sizes"])).to(device)
        self.q1 = Critic(observation_dim, action_dim, self.latent_dim, list(networks["critic_hidden_sizes"])).to(device); self.q2 = Critic(observation_dim, action_dim, self.latent_dim, list(networks["critic_hidden_sizes"])).to(device)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1).eval(), copy.deepcopy(self.q2).eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(sac["actor_lr"])); self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=float(sac["critic_lr"])); self.context_opt = torch.optim.Adam(self.context_encoder.parameters(), lr=float(sac["context_lr"]))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device); self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(sac["alpha_lr"])); self.target_entropy = -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor: return self.log_alpha.exp()
    def act(self, observation: torch.Tensor, z: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return self.actor.sample(observation, z, deterministic)[0]
    def prior(self, count: int = 1) -> tuple[torch.Tensor, torch.Tensor]: return self.context_encoder.prior(count, self.device)

    def context_tensor(self, context_by_task: list[list[list[Transition]]]) -> torch.Tensor:
        rows = []
        for episodes in context_by_task:
            rows.append(np.asarray([
                [np.concatenate([x.obs, x.action, [x.reward / self.reward_scale], x.next_obs, [float(x.terminated), float(x.truncated)]]) for x in transitions]
                for transitions in episodes
            ], dtype=np.float32))
        return torch.as_tensor(np.stack(rows), device=self.device)

    def infer_posterior(self, context_by_task: list[list[list[Transition]]]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(self.context_tensor(context_by_task))
    def sample_latent(self, mu: torch.Tensor, log_var: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return mu if deterministic else mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def update(self, context_by_task: list[list[list[Transition]]], rl_by_task: list[list[Transition]]) -> dict[str, float]:
        mu, log_var = self.infer_posterior(context_by_task); z = self.sample_latent(mu, log_var)
        batch_size, tasks = len(rl_by_task[0]), len(rl_by_task)
        obs = torch.as_tensor(np.concatenate([[x.obs for x in rows] for rows in rl_by_task]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.concatenate([[x.action for x in rows] for rows in rl_by_task]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.concatenate([[x.reward for x in rows] for rows in rl_by_task]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(np.concatenate([[x.next_obs for x in rows] for rows in rl_by_task]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.concatenate([[float(x.terminated) for x in rows] for rows in rl_by_task]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        expanded_z = z.repeat_interleave(batch_size, dim=0)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs, expanded_z.detach())
            target = rewards + self.gamma * (1 - done) * (torch.minimum(self.target_q1(next_obs, next_action, expanded_z.detach()), self.target_q2(next_obs, next_action, expanded_z.detach())) - self.alpha.detach() * next_logp)
        q_loss = nn.functional.mse_loss(self.q1(obs, actions, expanded_z), target) + nn.functional.mse_loss(self.q2(obs, actions, expanded_z), target)
        kl = self.context_encoder.kl_to_unit_normal(mu, log_var); encoder_loss = q_loss + self.kl_beta * kl
        self.q_opt.zero_grad(); self.context_opt.zero_grad(); encoder_loss.backward(); self.q_opt.step(); self.context_opt.step()
        actor_z = expanded_z.detach(); policy_action, logp = self.actor.sample(obs, actor_z); actor_loss = (self.alpha.detach() * logp - torch.minimum(self.q1(obs, policy_action, actor_z), self.q2(obs, policy_action, actor_z))).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean(); self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            for source, target_net in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(source.parameters(), target_net.parameters()): target_parameter.mul_(1 - self.tau).add_(parameter, alpha=self.tau)
        return {"q_loss": float(q_loss.detach()), "kl": float(kl.detach()), "actor_loss": float(actor_loss.detach()), "alpha": float(self.alpha.detach()), "posterior_variance": float(torch.exp(log_var).mean().detach())}

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for module in (self.context_encoder, self.actor, self.q1, self.q2, self.target_q1, self.target_q2):
            for tensor in module.state_dict().values(): digest.update(tensor.detach().cpu().numpy().tobytes())
        digest.update(self.log_alpha.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    def state_dict(self) -> dict[str, object]:
        return {
            "context_encoder": self.context_encoder.state_dict(), "actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_opt.state_dict(), "critic_optimizer": self.q_opt.state_dict(), "context_optimizer": self.context_opt.state_dict(), "alpha_optimizer": self.alpha_opt.state_dict(),
        }
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for name in ("context_encoder", "actor", "q1", "q2", "target_q1", "target_q2"): getattr(self, name).load_state_dict(state[name])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        optimizer_states = ((self.actor_opt, "actor_optimizer"), (self.q_opt, "critic_optimizer"), (self.context_opt, "context_optimizer"), (self.alpha_opt, "alpha_optimizer"))
        for optimizer, name in optimizer_states:
            if name not in state:
                raise ValueError(f"checkpoint lacks {name}; current checkpoints require resumable optimizer state")
            optimizer.load_state_dict(state[name])
