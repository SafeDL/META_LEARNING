"""PEARL-SAC implementation with prescribed gradient boundaries."""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .context_encoder import ContextEncoder
from .networks import Critic, GaussianActor
from .replay import Transition
from .task_representation import INTERACTION_OBSERVATION_INDEXES


class PEARLAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: Mapping[str, object], device: torch.device):
        self.device = device
        pearl = config["pearl"]
        networks = config["networks"]
        sac = config["sac"]
        self.observation_schema = str(config["environment"]["observation_schema"])
        self.action_dim = int(action_dim)
        self.latent_dim = int(pearl["latent_dim"])
        self.reward_scale = float(pearl["context_reward_scale"])
        self.gamma = float(sac["gamma"])
        self.tau = float(sac["tau"])
        self.kl_beta = float(pearl["kl_beta"])

        context_dim = observation_dim + action_dim + 1 + observation_dim + 2
        self.context_encoder = ContextEncoder(
            context_dim,
            self.latent_dim,
            list(networks["context_hidden_sizes"]),
        ).to(device)
        self.actor = GaussianActor(
            observation_dim,
            self.latent_dim,
            action_dim,
            list(networks["actor_hidden_sizes"]),
        ).to(device)
        critic_sizes = list(networks["critic_hidden_sizes"])
        self.q1 = Critic(observation_dim, action_dim, self.latent_dim, critic_sizes).to(device)
        self.q2 = Critic(observation_dim, action_dim, self.latent_dim, critic_sizes).to(device)
        self.target_q1 = copy.deepcopy(self.q1).eval()
        self.target_q2 = copy.deepcopy(self.q2).eval()

        representation = dict(config.get("task_representation", {}))
        self.disentangled = bool(representation.get("enabled", False))
        self.geometry_decoder: nn.Module | None = None
        self.interaction_decoder: nn.Module | None = None
        self.rule_decoder: nn.Module | None = None
        self.geometry_weight = self.interaction_weight = self.rule_weight = 0.0
        if self.disentangled:
            dims = tuple(int(value) for value in representation.get("latent_dims", (2, 2, 1)))
            if len(dims) != 3 or any(value < 1 for value in dims) or sum(dims) != self.latent_dim:
                raise ValueError("task_representation.latent_dims must be three positive values summing to latent_dim")
            self.geometry_dim, self.interaction_dim, self.rule_dim = dims
            (
                self.geometry_weight,
                self.interaction_weight,
                self.rule_weight,
            ) = (
                float(representation.get(name, 0.0))
                for name in ("geometry_weight", "interaction_weight", "rule_weight")
            )
            if min(self.geometry_weight, self.interaction_weight, self.rule_weight) < 0.0:
                raise ValueError("task representation auxiliary weights must be non-negative")
            hidden = max(16, self.latent_dim * 4)
            self.geometry_decoder = nn.Sequential(
                nn.Linear(self.geometry_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 5),
            ).to(device)
            self.interaction_decoder = nn.Sequential(
                nn.Linear(self.interaction_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
            ).to(device)
            self.rule_decoder = nn.Linear(self.rule_dim, 1).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(sac["actor_lr"]))
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self.q_opt = torch.optim.Adam(critic_parameters, lr=float(sac["critic_lr"]))
        context_parameters = list(self.context_encoder.parameters())
        if self.disentangled:
            context_parameters += list(self.geometry_decoder.parameters())
            context_parameters += list(self.interaction_decoder.parameters())
            context_parameters += list(self.rule_decoder.parameters())
        self.context_opt = torch.optim.Adam(context_parameters, lr=float(sac["context_lr"]))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(sac["alpha_lr"]))
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, observation: torch.Tensor, z: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return self.actor.sample(observation, z, deterministic)[0]

    def prior(self, count: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder.prior(count, self.device)

    def context_tensor(self, context_by_task: list[list[list[Transition]]]) -> torch.Tensor:
        rows: list[np.ndarray] = []
        for episodes in context_by_task:
            task_context = [
                [
                    np.concatenate(
                        [
                            transition.obs,
                            transition.action,
                            [transition.reward / self.reward_scale],
                            transition.next_obs,
                            [float(transition.terminated), float(transition.truncated)],
                        ]
                    )
                    for transition in episode
                ]
                for episode in episodes
            ]
            rows.append(np.asarray(task_context, dtype=np.float32))
        return torch.as_tensor(np.stack(rows), device=self.device)

    def infer_posterior(self, context_by_task: list[list[list[Transition]]]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(self.context_tensor(context_by_task))

    def sample_latent(self, mu: torch.Tensor, log_var: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def decode_task_representation(self, posterior_mean: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode the declared latent blocks for support-only semantic audits.

        This method never accepts a task identifier, taskbook field, or query
        label.  It merely exposes the auxiliary heads already used in training.
        """
        if not self.disentangled:
            raise ValueError("task-representation decoding requires --disentangled-representation")
        if posterior_mean.ndim != 2 or posterior_mean.shape[-1] != self.latent_dim:
            raise ValueError("posterior_mean must have shape [tasks, latent_dim]")
        geometry_z, interaction_z, rule_z = torch.split(
            posterior_mean,
            (self.geometry_dim, self.interaction_dim, self.rule_dim),
            dim=-1,
        )
        rule_logit = self.rule_decoder(rule_z)
        return {
            "geometry": self.geometry_decoder(geometry_z),
            "interaction": self.interaction_decoder(interaction_z),
            "entry_order_logit": rule_logit,
            "entry_order_probability": torch.sigmoid(rule_logit),
        }

    def _auxiliary_loss(
        self,
        mu: torch.Tensor,
        context: torch.Tensor,
        targets: list[Mapping[str, Any]] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=self.device)
        if not self.disentangled:
            return zero, {
                "geometry_aux_loss": zero,
                "interaction_aux_loss": zero,
                "rule_aux_loss": zero,
            }
        if targets is None or len(targets) != len(mu):
            raise ValueError("disentangled PEARL update requires one semantic target per sampled task")
        geometry_target = torch.as_tensor(
            np.stack([np.asarray(target["geometry"], dtype=np.float32) for target in targets]),
            device=self.device,
        )
        rule_target = torch.as_tensor(
            np.asarray([float(np.asarray(target["entry_order"])) for target in targets], dtype=np.float32),
            device=self.device,
        ).unsqueeze(-1)
        decoded = self.decode_task_representation(mu)
        # These observation fields are label-free and available in every
        # support transition: arrival-time difference, relative speed, TTC.
        interaction_target = context[..., list(INTERACTION_OBSERVATION_INDEXES)].mean(dim=(1, 2))
        geometry_loss = nn.functional.mse_loss(decoded["geometry"], geometry_target)
        interaction_loss = nn.functional.mse_loss(decoded["interaction"], interaction_target)
        rule_loss = nn.functional.binary_cross_entropy_with_logits(decoded["entry_order_logit"], rule_target)
        weighted = (
            self.geometry_weight * geometry_loss
            + self.interaction_weight * interaction_loss
            + self.rule_weight * rule_loss
        )
        return weighted, {
            "geometry_aux_loss": geometry_loss,
            "interaction_aux_loss": interaction_loss,
            "rule_aux_loss": rule_loss,
        }

    def update(
        self,
        context_by_task: list[list[list[Transition]]],
        rl_by_task: list[list[Transition]],
        task_targets: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        context = self.context_tensor(context_by_task)
        mu, log_var = self.context_encoder(context)
        z = self.sample_latent(mu, log_var)
        batch_size = len(rl_by_task[0])
        transitions = [transition for task_rows in rl_by_task for transition in task_rows]
        obs = torch.as_tensor(
            np.asarray([transition.obs for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray([transition.action for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            np.asarray([transition.reward for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        next_obs = torch.as_tensor(
            np.asarray([transition.next_obs for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.as_tensor(
            np.asarray([float(transition.terminated) for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        expanded_z = z.repeat_interleave(batch_size, dim=0)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs, expanded_z.detach())
            next_value = torch.minimum(
                self.target_q1(next_obs, next_action, expanded_z.detach()),
                self.target_q2(next_obs, next_action, expanded_z.detach()),
            ) - self.alpha.detach() * next_logp
            target = rewards + self.gamma * (1 - done) * next_value
        q_loss = (
            nn.functional.mse_loss(self.q1(obs, actions, expanded_z), target)
            + nn.functional.mse_loss(self.q2(obs, actions, expanded_z), target)
        )
        kl = self.context_encoder.kl_to_unit_normal(mu, log_var)
        auxiliary, auxiliary_metrics = self._auxiliary_loss(mu, context, task_targets)
        encoder_loss = q_loss + self.kl_beta * kl + auxiliary
        self.q_opt.zero_grad()
        self.context_opt.zero_grad()
        encoder_loss.backward()
        self.q_opt.step()
        self.context_opt.step()

        actor_z = expanded_z.detach()
        policy_action, logp = self.actor.sample(obs, actor_z)
        actor_loss = (
            self.alpha.detach() * logp
            - torch.minimum(
                self.q1(obs, policy_action, actor_z),
                self.q2(obs, policy_action, actor_z),
            )
        ).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for source, target_net in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(source.parameters(), target_net.parameters()):
                    target_parameter.mul_(1 - self.tau).add_(parameter, alpha=self.tau)
        return {
            "q_loss": float(q_loss.detach()),
            "kl": float(kl.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "posterior_variance": float(torch.exp(log_var).mean().detach()),
            "auxiliary_loss": float(auxiliary.detach()),
            **{name: float(value.detach()) for name, value in auxiliary_metrics.items()},
        }

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        modules = [self.context_encoder, self.actor, self.q1, self.q2, self.target_q1, self.target_q2]
        if self.disentangled:
            modules += [self.geometry_decoder, self.interaction_decoder, self.rule_decoder]
        for module in modules:
            for tensor in module.state_dict().values():
                digest.update(tensor.detach().cpu().numpy().tobytes())
        digest.update(self.log_alpha.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, object]:
        result = {
            "context_encoder": self.context_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.q_opt.state_dict(),
            "context_optimizer": self.context_opt.state_dict(),
            "alpha_optimizer": self.alpha_opt.state_dict(),
        }
        if self.disentangled:
            result["task_representation"] = {
                "geometry_decoder": self.geometry_decoder.state_dict(),
                "interaction_decoder": self.interaction_decoder.state_dict(),
                "rule_decoder": self.rule_decoder.state_dict(),
            }
        return result

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for name in ("context_encoder", "actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(state[name])
        representation_state = state.get("task_representation")
        if self.disentangled:
            if not isinstance(representation_state, Mapping):
                raise ValueError("checkpoint lacks disentangled task-representation state")
            self.geometry_decoder.load_state_dict(representation_state["geometry_decoder"])
            self.interaction_decoder.load_state_dict(representation_state["interaction_decoder"])
            self.rule_decoder.load_state_dict(representation_state["rule_decoder"])
        elif representation_state is not None:
            raise ValueError("checkpoint uses a disentangled task representation; enable it in the configuration")
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        optimizer_states = (
            (self.actor_opt, "actor_optimizer"),
            (self.q_opt, "critic_optimizer"),
            (self.context_opt, "context_optimizer"),
            (self.alpha_opt, "alpha_optimizer"),
        )
        for optimizer, name in optimizer_states:
            if name not in state:
                raise ValueError(f"checkpoint lacks {name}; current checkpoints require resumable optimizer state")
            optimizer.load_state_dict(state[name])
