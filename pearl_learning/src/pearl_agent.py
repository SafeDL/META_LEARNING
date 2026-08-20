"""PEARL-SAC implementation with prescribed gradient boundaries."""
from __future__ import annotations

import copy
import hashlib
from itertools import chain
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .context_encoder import ContextEncoder, kl_diag_normal
from .moe import (
    DESCRIPTOR_FIELDS,
    DESCRIPTOR_SCHEMA,
    PhysicalTaskDescriptor,
    PosteriorRoutedMoEActor,
    PosteriorRouter,
    RouteContext,
    RoutingOutput,
    load_balance_loss,
    intervene_route as build_intervened_route,
    route_context,
)
from .networks import Critic, GaussianActor, LatentFiLMCritic, LatentGammaOnlyFiLMCritic
from .replay import Transition
from .task_representation import INTERACTION_OBSERVATION_INDEXES
from .scenario_encoder import (
    DESCRIPTOR_FIELDS as SCENARIO_DESCRIPTOR_FIELDS,
    DESCRIPTOR_SCHEMA as SCENARIO_DESCRIPTOR_SCHEMA,
    ScenarioConditionedPrior,
    ScenarioEncoder,
    build_task_descriptor,
)


class PEARLAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: Mapping[str, object], device: torch.device):
        self.device = device
        pearl = config["pearl"]
        networks = config["networks"]
        sac = config["sac"]
        self.observation_schema = str(config["environment"]["observation_schema"])
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(pearl["latent_dim"])
        self.context_aggregation = str(pearl["context_aggregation"])
        self.reward_scale = float(pearl["context_reward_scale"])
        self.critic_reward_scale = float(pearl.get("critic_reward_scale", 1.0))
        if self.reward_scale <= 0.0 or self.critic_reward_scale <= 0.0:
            raise ValueError("PEARL context and critic reward scales must be positive")
        self.gamma = float(sac["gamma"])
        self.tau = float(sac["tau"])
        self.kl_beta = float(pearl["kl_beta"])
        self.no_context_training = bool(config.get("ablation", {}).get("no_context_training", False))
        representation_cfg = dict(config.get("scenario_representation", {}))
        prior_cfg = dict(config.get("scenario_prior", {}))
        self.scenario_representation_enabled = bool(representation_cfg.get("enabled", False))
        self.scenario_prior_mode = str(prior_cfg.get("mode", "unit_normal"))
        if self.scenario_prior_mode not in {"unit_normal", "task_conditioned"}:
            raise ValueError("scenario_prior.mode must be unit_normal or task_conditioned")
        if self.scenario_prior_mode == "task_conditioned" and not self.scenario_representation_enabled:
            raise ValueError("task_conditioned scenario prior requires scenario_representation.enabled")
        self.scenario_encoder: ScenarioEncoder | None = None
        self.scenario_prior: ScenarioConditionedPrior | None = None
        self.scenario_embedding_dim: int | None = None
        if self.scenario_representation_enabled:
            self.scenario_embedding_dim = int(representation_cfg.get("embedding_dim", 8))
            self.scenario_encoder = ScenarioEncoder(
                self.scenario_embedding_dim, [int(v) for v in representation_cfg.get("hidden_sizes", [32, 16])]
            ).to(device)
            if self.scenario_prior_mode == "task_conditioned":
                self.scenario_prior = ScenarioConditionedPrior(
                    self.scenario_embedding_dim, self.latent_dim, [int(v) for v in prior_cfg.get("hidden_sizes", [32])]
                ).to(device)

        context_dim = observation_dim + action_dim + 1 + observation_dim + 2
        self.context_encoder = ContextEncoder(
            context_dim,
            self.latent_dim,
            list(networks["context_hidden_sizes"]),
            self.context_aggregation,
        ).to(device)
        if self.no_context_training:
            self.context_encoder.requires_grad_(False)
        self.actor_architecture = str(networks.get("actor_architecture", "dense"))
        self.actor_hidden_sizes = [int(value) for value in networks["actor_hidden_sizes"]]
        self.critic_hidden_sizes = [int(value) for value in networks["critic_hidden_sizes"]]
        self.context_hidden_sizes = [int(value) for value in networks["context_hidden_sizes"]]
        self.router_hidden_sizes: list[int] | None = None
        self.expert_hidden_size: int | None = None
        self.router: PosteriorRouter | None = None
        self.descriptor_schema: str | None = None
        self.num_experts: int | None = None
        self.top_k: int | None = None
        self.routing: str | None = None
        self.router_input_mode: str | None = None
        self.load_balance_weight = 0.0
        if self.actor_architecture == "dense":
            self.actor = GaussianActor(
                observation_dim,
                self.latent_dim,
                action_dim,
                self.actor_hidden_sizes,
            ).to(device)
        elif self.actor_architecture == "posterior_routed_moe":
            moe = dict(networks.get("moe", {}))
            self.descriptor_schema = str(moe.get("descriptor_schema", ""))
            if self.descriptor_schema != DESCRIPTOR_SCHEMA:
                raise ValueError(f"unsupported MoE descriptor schema: {self.descriptor_schema!r}")
            normalization = dict(moe.get("descriptor_normalization", {}))
            if tuple(normalization) != DESCRIPTOR_FIELDS:
                raise ValueError("MoE descriptor_normalization must exactly follow the descriptor field order")
            self.num_experts = int(moe["num_experts"])
            self.top_k = int(moe["top_k"])
            self.routing = str(moe["routing"])
            self.router_input_mode = str(
                moe.get("input_mode", "static_posterior_mean_logvar")
            )
            self.load_balance_weight = float(moe["load_balance_weight"])
            self.router_hidden_sizes = [int(value) for value in moe["router_hidden_sizes"]]
            self.expert_hidden_size = int(moe["expert_hidden_size"])
            if self.load_balance_weight < 0.0:
                raise ValueError("load_balance_weight must be non-negative")
            self.router = PosteriorRouter(
                len(DESCRIPTOR_FIELDS),
                self.latent_dim,
                self.num_experts,
                self.top_k,
                self.routing,
                self.router_hidden_sizes,
                self.router_input_mode,
            ).to(device)
            self.actor = PosteriorRoutedMoEActor(
                observation_dim,
                self.latent_dim,
                action_dim,
                self.actor_hidden_sizes,
                self.num_experts,
                self.expert_hidden_size,
            ).to(device)
        else:
            raise ValueError(f"unsupported actor_architecture: {self.actor_architecture!r}")
        critic_sizes = list(networks["critic_hidden_sizes"])
        self.critic_architecture = str(networks.get("critic_architecture", "dense"))
        critic_classes = {
            "dense": Critic,
            "latent_film_dense": LatentFiLMCritic,
            "latent_film_gamma_only": LatentGammaOnlyFiLMCritic,
        }
        if self.critic_architecture not in critic_classes:
            raise ValueError(f"unsupported critic_architecture: {self.critic_architecture!r}")
        critic_class = critic_classes[self.critic_architecture]
        self.q1 = critic_class(observation_dim, action_dim, self.latent_dim, critic_sizes).to(device)
        self.q2 = critic_class(observation_dim, action_dim, self.latent_dim, critic_sizes).to(device)
        self.target_q1 = copy.deepcopy(self.q1).eval()
        self.target_q2 = copy.deepcopy(self.q2).eval()

        representation = dict(config.get("task_representation", {}))
        self.disentangled = bool(representation.get("enabled", False))
        if self.no_context_training and self.disentangled:
            raise ValueError("no-context training cannot enable disentangled posterior supervision")
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

        actor_parameters = self.actor.parameters() if self.router is None else chain(
            self.actor.parameters(), self.router.parameters()
        )
        self.actor_opt = torch.optim.Adam(actor_parameters, lr=float(sac["actor_lr"]))
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self.q_opt = torch.optim.Adam(critic_parameters, lr=float(sac["critic_lr"]))
        context_parameters = list(self.context_encoder.parameters())
        if self.scenario_encoder is not None:
            context_parameters += list(self.scenario_encoder.parameters())
        if self.scenario_prior is not None:
            context_parameters += list(self.scenario_prior.parameters())
        if self.disentangled:
            context_parameters += list(self.geometry_decoder.parameters())
            context_parameters += list(self.interaction_decoder.parameters())
            context_parameters += list(self.rule_decoder.parameters())
        self.context_opt = torch.optim.Adam(context_parameters, lr=float(sac["context_lr"]))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(sac["alpha_lr"]))
        self.target_entropy = -float(action_dim)
        self.last_router_audits: list[dict[str, Any]] = []

    @property
    def alpha(self) -> torch.Tensor:
        # A positive reward rescaling must also rescale entropy regularization
        # to preserve the SAC objective and optimal policy. ``log_alpha``
        # remains the dimensionless automatically tuned temperature.
        return self.critic_reward_scale * self.log_alpha.exp()

    def act(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool,
        route: RouteContext | None = None,
    ) -> torch.Tensor:
        if self.actor_architecture == "dense":
            if route is not None:
                raise ValueError("dense actor does not accept a route context")
            return self.actor.sample(observation, z, deterministic)[0]
        if route is None:
            raise ValueError("posterior-routed MoE actor requires an explicit task-level route context")
        weights = route.weight_tensor(self.device).expand(len(observation), -1)
        return self.actor.sample(observation, z, weights, deterministic)[0]

    def compute_route(
        self,
        descriptor: PhysicalTaskDescriptor,
        posterior_mean: torch.Tensor,
        posterior_log_variance: torch.Tensor,
        posterior_version: int,
        *,
        gradient_enabled: bool = False,
    ) -> RouteContext | None:
        """Compute one task route; collection callers cache it for the episode."""
        if self.actor_architecture == "dense":
            return None
        if descriptor.schema != self.descriptor_schema or descriptor.fields != DESCRIPTOR_FIELDS:
            raise ValueError("physical task descriptor is incompatible with the configured router")
        if posterior_mean.shape != (1, self.latent_dim) or posterior_log_variance.shape != (1, self.latent_dim):
            raise ValueError("a collection route requires posterior tensors with shape [1, latent_dim]")
        descriptor_tensor = descriptor.tensor(self.device).unsqueeze(0)
        with torch.set_grad_enabled(gradient_enabled):
            output = self.router(descriptor_tensor, posterior_mean, posterior_log_variance)
        return route_context(
            descriptor,
            posterior_version,
            posterior_mean,
            posterior_log_variance,
            output,
            gradient_enabled=gradient_enabled,
        )

    def intervene_route(
        self,
        source: RouteContext,
        *,
        posterior_version: int,
        mode: str,
        expert_index: int | None = None,
    ) -> RouteContext:
        if self.actor_architecture != "posterior_routed_moe":
            raise ValueError("route interventions require a MoE actor")
        if mode == "frozen_prior":
            weights = source.weights
        elif mode == "uniform":
            weights = [1.0 / self.num_experts] * self.num_experts
        elif mode == "expert_knockout":
            if expert_index is None or not 0 <= int(expert_index) < self.num_experts:
                raise ValueError("expert_knockout requires a valid expert_index")
            weights = list(source.weights)
            weights[int(expert_index)] = 0.0
            if sum(weights) <= 0.0:
                raise ValueError("cannot knock out the only active expert")
        else:
            raise ValueError(f"unsupported route intervention: {mode!r}")
        return build_intervened_route(
            source,
            weights,
            posterior_version=posterior_version,
            intervention=mode if expert_index is None else f"{mode}:{expert_index}",
        )

    def expert_action_means(
        self,
        observation: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.actor_architecture != "posterior_routed_moe":
            raise ValueError("expert action audit requires a MoE actor")
        return self.actor.expert_action_means(observation, latent)

    def _scenario_prior(self, tasks: list[Any] | None, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scenario_prior_mode == "unit_normal":
            return self.context_encoder.prior(count, self.device)
        if tasks is None or len(tasks) != count:
            raise ValueError("task-conditioned prior requires one task per posterior row")
        descriptors = torch.as_tensor(
            np.stack([build_task_descriptor(task) for task in tasks]), dtype=torch.float32, device=self.device
        )
        return self.scenario_prior(self.scenario_encoder(descriptors))

    def prior(self, count: int = 1, tasks: list[Any] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._scenario_prior(tasks, count)

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

    def infer_posterior(self, context_by_task: list[list[list[Transition]]], tasks: list[Any] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        prior = self._scenario_prior(tasks, len(context_by_task))
        if self.scenario_prior_mode == "unit_normal":
            return self.context_encoder(self.context_tensor(context_by_task))
        return self.context_encoder(self.context_tensor(context_by_task), prior)

    def sample_latent(self, mu: torch.Tensor, log_var: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def sample_latent_seeded(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Sample a posterior latent without consuming the training RNG stream."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device="cpu").to(self.device)
        return mu + noise * torch.exp(0.5 * log_var)

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
        task_descriptors: list[PhysicalTaskDescriptor] | None = None,
        posterior_versions: list[int] | None = None,
        scenario_tasks: list[Any] | None = None,
    ) -> dict[str, float]:
        context = None
        if self.no_context_training:
            mu, log_var = self.prior(len(context_by_task), scenario_tasks)
            z = mu
        else:
            context = self.context_tensor(context_by_task)
            prior_mu, prior_log_var = self._scenario_prior(scenario_tasks, len(context_by_task))
            mu, log_var = self.context_encoder(context, (prior_mu, prior_log_var) if self.scenario_prior_mode == "task_conditioned" else None)
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
        rewards = self.critic_reward_scale * torch.as_tensor(
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
            # Time-limit truncation is not an MDP terminal: bootstrap through
            # it, while physical/rule terminations stop the Bellman target.
            np.asarray([float(transition.terminated) for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        expanded_z = z.repeat_interleave(batch_size, dim=0)
        training_route: RoutingOutput | None = None
        descriptor_tensor: torch.Tensor | None = None
        if self.actor_architecture == "posterior_routed_moe":
            if task_descriptors is None or len(task_descriptors) != len(context_by_task):
                raise ValueError("MoE update requires one physical descriptor per sampled task")
            if any(
                descriptor.schema != self.descriptor_schema or descriptor.fields != DESCRIPTOR_FIELDS
                for descriptor in task_descriptors
            ):
                raise ValueError("MoE update received an incompatible physical task descriptor")
            descriptor_tensor = torch.stack(
                [descriptor.tensor(self.device) for descriptor in task_descriptors]
            )
        hashes_before_critic = self.module_hashes()
        actor_hashes_before_critic = {
            name: hashes_before_critic[name]
            for name in (("actor", "router") if self.router is not None else ("actor",))
        }
        with torch.no_grad():
            if self.actor_architecture == "dense":
                next_action, next_logp = self.actor.sample(next_obs, expanded_z.detach())
            else:
                target_route = self.router(descriptor_tensor, mu.detach(), log_var.detach())
                target_weights = target_route.weights.repeat_interleave(batch_size, dim=0)
                next_action, next_logp = self.actor.sample(
                    next_obs,
                    expanded_z.detach(),
                    target_weights,
                )
            next_value = torch.minimum(
                self.target_q1(next_obs, next_action, expanded_z.detach()),
                self.target_q2(next_obs, next_action, expanded_z.detach()),
            ) - self.alpha.detach() * next_logp
            target = rewards + self.gamma * (1 - done) * next_value
        q_loss = (
            nn.functional.mse_loss(self.q1(obs, actions, expanded_z), target)
            + nn.functional.mse_loss(self.q2(obs, actions, expanded_z), target)
        )
        if self.scenario_prior_mode == "task_conditioned":
            kl = kl_diag_normal(mu, log_var, prior_mu, prior_log_var)
        else:
            kl = self.context_encoder.kl_to_unit_normal(mu, log_var)
        zero = torch.zeros((), device=self.device)
        if self.no_context_training:
            auxiliary = zero
            auxiliary_metrics = {
                "geometry_aux_loss": zero,
                "interaction_aux_loss": zero,
                "rule_aux_loss": zero,
            }
            encoder_loss = q_loss
        else:
            auxiliary, auxiliary_metrics = self._auxiliary_loss(mu, context, task_targets)
            encoder_loss = q_loss + self.kl_beta * kl + auxiliary
        self.q_opt.zero_grad()
        self.context_opt.zero_grad()
        # How strongly the Bellman loss responds to the latent itself.  A
        # Critic that is conditioning-insensitive keeps this near zero even
        # while the encoder posterior separates tasks; the FiLM-critic round
        # uses it to confirm task-dependent Q formation during training.
        if z.grad_fn is None:
            critic_latent_gradient_norm = 0.0
        else:
            critic_latent_grads = torch.autograd.grad(q_loss, z, retain_graph=True, allow_unused=True)[0]
            critic_latent_gradient_norm = (
                0.0
                if critic_latent_grads is None
                else float(critic_latent_grads.detach().norm(dim=-1).mean())
            )
        if self.no_context_training:
            encoder_critic_gradient_norm = 0.0
            posterior_prior_mean_l2 = 0.0
            evidence_to_prior_precision_ratio = 0.0
        else:
            # Log-only diagnostics for the Gate 3 causal chain.  The
            # actor-side encoder gradient is ~0 by design (actor_z.detach()),
            # so the critic gradient is the only signal that the encoder is
            # actually optimized through the Bellman objective.  retain_graph
            # keeps the subsequent update backward byte-identical.
            encoder_critic_grads = torch.autograd.grad(
                q_loss,
                list(self.context_encoder.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            encoder_gradient_squares = [
                grad.detach().square().sum() for grad in encoder_critic_grads if grad is not None
            ]
            encoder_critic_gradient_norm = (
                0.0
                if not encoder_gradient_squares
                else float(torch.sqrt(torch.stack(encoder_gradient_squares).sum()).detach())
            )
            posterior_prior_mean_l2 = float((mu - prior_mu).norm(dim=-1).mean().detach())
            posterior_precision = torch.exp(-log_var)
            prior_precision = torch.exp(-prior_log_var)
            evidence_to_prior_precision_ratio = float(
                ((posterior_precision - prior_precision).mean()
                 / (prior_precision.mean() + 1e-8)).detach()
            )
        encoder_loss.backward()
        self.q_opt.step()
        if not self.no_context_training:
            self.context_opt.step()
        hashes_after_critic = self.module_hashes()
        critic_phase_actor_unchanged = all(
            hashes_after_critic[name] == value
            for name, value in actor_hashes_before_critic.items()
        )
        critic_hashes_before_actor = {
            name: hashes_after_critic[name] for name in ("q1", "q2")
        }

        # Remove gradients from the critic/encoder phase so the actor boundary
        # is directly auditable instead of being obscured by stale gradients.
        self.q_opt.zero_grad(set_to_none=True)
        self.context_opt.zero_grad(set_to_none=True)

        actor_z = expanded_z.detach()
        if self.actor_architecture == "dense":
            policy_action, logp = self.actor.sample(obs, actor_z)
            balance = zero
        else:
            training_route = self.router(descriptor_tensor, mu.detach(), log_var.detach())
            actor_weights = training_route.weights.repeat_interleave(batch_size, dim=0)
            policy_action, logp = self.actor.sample(obs, actor_z, actor_weights)
            balance = load_balance_loss(training_route.weights)
        for critic in (self.q1, self.q2):
            critic.requires_grad_(False)
        actor_main_loss = (
            self.alpha.detach() * logp
            - torch.minimum(
                self.q1(obs, policy_action, actor_z),
                self.q2(obs, policy_action, actor_z),
            )
        ).mean()
        actor_total_loss = actor_main_loss + self.load_balance_weight * balance
        self.actor_opt.zero_grad(set_to_none=True)
        actor_total_loss.backward()
        actor_gradients = self._actor_gradient_audit()
        self.actor_opt.step()
        hashes_after_actor = self.module_hashes()
        actor_phase_critic_unchanged = all(
            hashes_after_actor[name] == value
            for name, value in critic_hashes_before_actor.items()
        )
        for critic in (self.q1, self.q2):
            critic.requires_grad_(True)
        self.last_router_audits = []
        if training_route is not None:
            versions = posterior_versions or [0] * len(context_by_task)
            if len(versions) != len(context_by_task):
                raise ValueError("posterior_versions must match the task batch")
            routing_metrics = self._routing_metrics(training_route)
            hashes = self.module_hashes()
            for index, descriptor in enumerate(task_descriptors):
                single_output = RoutingOutput(
                    training_route.logits[index:index + 1],
                    training_route.soft_weights[index:index + 1],
                    training_route.top_k_mask[index:index + 1],
                    training_route.weights[index:index + 1],
                    training_route.entropy[index:index + 1],
                    training_route.top_k_indexes[index:index + 1],
                )
                audit = route_context(
                    descriptor,
                    int(versions[index]),
                    mu[index:index + 1],
                    log_var[index:index + 1],
                    single_output,
                    gradient_enabled=True,
                ).audit_dict()
                audit.update({
                    "actor_main_loss": float(actor_main_loss.detach()),
                    "balance_loss": float(balance.detach()),
                    "expert_load": [
                        routing_metrics[f"expert_{expert}_load"]
                        for expert in range(self.num_experts)
                    ],
                    "expert_load_cv": routing_metrics["expert_load_cv"],
                    "gradient_norms": dict(actor_gradients),
                    "module_hashes_after_update": hashes,
                    "parameter_hash_after_update": self.parameter_hash(),
                })
                self.last_router_audits.append(audit)

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
            "actor_loss": float(actor_main_loss.detach()),
            "actor_total_loss": float(actor_total_loss.detach()),
            "balance_loss": float(balance.detach()),
            "critic_phase_actor_unchanged": float(critic_phase_actor_unchanged),
            "actor_phase_critic_unchanged": float(actor_phase_critic_unchanged),
            "alpha": float(self.alpha.detach()),
            "posterior_variance": float(torch.exp(log_var).mean().detach()),
            "critic_reward_scale": self.critic_reward_scale,
            "auxiliary_loss": float(auxiliary.detach()),
            "context_encoder_critic_gradient_norm": encoder_critic_gradient_norm,
            "posterior_prior_mean_l2": posterior_prior_mean_l2,
            "evidence_to_prior_precision_ratio": evidence_to_prior_precision_ratio,
            "critic_latent_gradient_norm": critic_latent_gradient_norm,
            **actor_gradients,
            **self._routing_metrics(training_route),
            **{name: float(value.detach()) for name, value in auxiliary_metrics.items()},
        }

    @staticmethod
    def _gradient_norm(parameters: Any) -> float:
        squares = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
        return 0.0 if not squares else float(torch.sqrt(torch.stack(squares).sum()).detach())

    def _actor_gradient_audit(self) -> dict[str, float]:
        result = {
            "actor_shared_gradient_norm": self._gradient_norm(
                self.actor.parameters() if self.actor_architecture == "dense" else self.actor.shared_trunk.parameters()
            ),
            "context_encoder_actor_gradient_norm": self._gradient_norm(self.context_encoder.parameters()),
            "critic_actor_gradient_norm": self._gradient_norm(chain(self.q1.parameters(), self.q2.parameters())),
        }
        if self.actor_architecture == "posterior_routed_moe":
            result["router_gradient_norm"] = self._gradient_norm(self.router.parameters())
            result["actor_head_gradient_norm"] = self._gradient_norm(self.actor.gaussian_head.parameters())
            for index, expert in enumerate(self.actor.residual_experts):
                result[f"expert_{index}_gradient_norm"] = self._gradient_norm(expert.parameters())
        return result

    def _routing_metrics(self, output: RoutingOutput | None) -> dict[str, float]:
        if output is None:
            return {}
        load = output.weights.detach().mean(dim=0)
        mean = load.mean()
        cv = load.std(unbiased=False) / mean.clamp_min(1e-12)
        metrics = {
            "router_entropy": float(output.entropy.detach().mean()),
            "expert_load_cv": float(cv),
        }
        metrics.update({f"expert_{index}_load": float(value) for index, value in enumerate(load)})
        return metrics

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for value in self.module_hashes().values():
            digest.update(value.encode("ascii"))
        return digest.hexdigest()

    def module_hashes(self) -> dict[str, str]:
        modules = {
            "context_encoder": self.context_encoder,
            "actor": self.actor,
            "q1": self.q1,
            "q2": self.q2,
            "target_q1": self.target_q1,
            "target_q2": self.target_q2,
        }
        if self.scenario_encoder is not None:
            modules["scenario_encoder"] = self.scenario_encoder
        if self.scenario_prior is not None:
            modules["scenario_prior"] = self.scenario_prior
        if self.disentangled:
            modules.update({
                "geometry_decoder": self.geometry_decoder,
                "interaction_decoder": self.interaction_decoder,
                "rule_decoder": self.rule_decoder,
            })
        if self.router is not None:
            modules["router"] = self.router
        result: dict[str, str] = {}
        for name, module in modules.items():
            digest = hashlib.sha256()
            for tensor in module.state_dict().values():
                digest.update(tensor.detach().cpu().numpy().tobytes())
            result[name] = digest.hexdigest()
        alpha_digest = hashlib.sha256(self.log_alpha.detach().cpu().numpy().tobytes())
        result["log_alpha"] = alpha_digest.hexdigest()
        return result

    def architecture_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema": "pearl_actor_architecture",
            "actor_architecture": self.actor_architecture,
            "observation_schema": self.observation_schema,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "context_aggregation": self.context_aggregation,
            "actor_hidden_sizes": self.actor_hidden_sizes,
            "critic_hidden_sizes": self.critic_hidden_sizes,
            "critic_architecture": self.critic_architecture,
            "context_hidden_sizes": self.context_hidden_sizes,
            "scenario_representation": {
                "enabled": self.scenario_representation_enabled,
                "descriptor_schema": SCENARIO_DESCRIPTOR_SCHEMA if self.scenario_representation_enabled else None,
                "descriptor_fields": list(SCENARIO_DESCRIPTOR_FIELDS) if self.scenario_representation_enabled else [],
                "embedding_dim": self.scenario_embedding_dim,
                "prior_mode": self.scenario_prior_mode,
            },
        }
        if self.actor_architecture == "posterior_routed_moe":
            metadata["moe"] = {
                "descriptor_schema": self.descriptor_schema,
                "descriptor_fields": list(DESCRIPTOR_FIELDS),
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "routing": self.routing,
                "router_input_fields": list(self.router.input_fields),
                "load_balance_weight": self.load_balance_weight,
                "router_hidden_sizes": self.router_hidden_sizes,
                "expert_hidden_size": self.expert_hidden_size,
                "actor_form": "shared_trunk_plus_weighted_residual_experts_plus_unified_gaussian_head",
                "critic_architecture": "dense_twin_critics",
                "posterior_gradient_boundary": "router_receives_detached_mean_and_log_variance",
            }
        return metadata

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
            "architecture_metadata": self.architecture_metadata(),
        }
        if self.router is not None:
            result["router"] = self.router.state_dict()
        if self.scenario_encoder is not None:
            result["scenario_encoder"] = self.scenario_encoder.state_dict()
        if self.scenario_prior is not None:
            result["scenario_prior"] = self.scenario_prior.state_dict()
        if self.disentangled:
            result["task_representation"] = {
                "geometry_decoder": self.geometry_decoder.state_dict(),
                "interaction_decoder": self.interaction_decoder.state_dict(),
                "rule_decoder": self.rule_decoder.state_dict(),
            }
        return result

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        stored_metadata = dict(state.get("architecture_metadata") or {})
        # Checkpoints saved before critic architectures existed always used
        # the dense twin critic; default the missing key instead of breaking
        # their load path.
        stored_metadata.setdefault("critic_architecture", "dense")
        if stored_metadata != self.architecture_metadata():
            raise ValueError("checkpoint agent state has incompatible architecture metadata")
        for name in ("context_encoder", "actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(state[name])
        if self.router is not None:
            if "router" not in state:
                raise ValueError("MoE checkpoint lacks router state")
            self.router.load_state_dict(state["router"])
        elif "router" in state:
            raise ValueError("dense agent cannot load MoE router state")
        if self.scenario_encoder is not None:
            if "scenario_encoder" not in state:
                raise ValueError("checkpoint lacks scenario encoder state")
            self.scenario_encoder.load_state_dict(state["scenario_encoder"])
        elif "scenario_encoder" in state:
            raise ValueError("checkpoint uses a scenario encoder; enable it in the configuration")
        if self.scenario_prior is not None:
            if "scenario_prior" not in state:
                raise ValueError("checkpoint lacks scenario prior state")
            self.scenario_prior.load_state_dict(state["scenario_prior"])
        elif "scenario_prior" in state:
            raise ValueError("checkpoint uses a task-conditioned scenario prior; enable it in the configuration")
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
