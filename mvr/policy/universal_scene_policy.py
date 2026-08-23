"""Universal MoE policy over interaction candidates and conflict-relative x0."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .moe_router import TaskAwareMoERouter


@dataclass(frozen=True)
class UniversalSceneAction:
    expert_index: torch.Tensor
    candidate_index: torch.Tensor
    continuous: torch.Tensor
    option_index: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class _InitialConditionExpert(nn.Module):
    def __init__(self, input_dim: int, continuous_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, continuous_dim)
        self.log_std = nn.Parameter(torch.zeros(continuous_dim))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.network(inputs)
        mean = self.mean(features)
        return mean, self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)


class UniversalScenePolicy(nn.Module):
    def __init__(self, scene_dim: int, latent_dim: int, continuous_dim: int, option_count: int, num_experts: int) -> None:
        super().__init__()
        inputs = scene_dim + latent_dim
        self.router = TaskAwareMoERouter(scene_dim, latent_dim, num_experts)
        self.experts = nn.ModuleList(_InitialConditionExpert(inputs, continuous_dim) for _ in range(num_experts))
        self.candidate_scorer = nn.Sequential(nn.Linear(2 * scene_dim + latent_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.option_head = nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, option_count))
        self.value_head = nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 1))
        self.continuous_dim = continuous_dim

    @staticmethod
    def _batch(value: torch.Tensor, dimensions: int) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == dimensions - 1 else value

    def _distributions(
        self,
        scene_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[Categorical, Categorical, list[Normal], Categorical, torch.Tensor]:
        scene_embedding = self._batch(scene_embedding, 2)
        latent = self._batch(latent, 2)
        candidate_embeddings = self._batch(candidate_embeddings, 3)
        candidate_mask = self._batch(candidate_mask, 2).bool()
        if candidate_embeddings.shape[:2] != candidate_mask.shape or candidate_embeddings.shape[0] != scene_embedding.shape[0]:
            raise ValueError("candidate embeddings and masks must align with scene batch")
        inputs = torch.cat((scene_embedding, latent), dim=-1)
        router = Categorical(logits=self.router(scene_embedding, latent))
        expanded_scene = scene_embedding[:, None, :].expand(-1, candidate_embeddings.shape[1], -1)
        expanded_latent = latent[:, None, :].expand(-1, candidate_embeddings.shape[1], -1)
        candidate_logits = self.candidate_scorer(
            torch.cat((candidate_embeddings, expanded_scene, expanded_latent), dim=-1)
        ).squeeze(-1).masked_fill(~candidate_mask, -torch.inf)
        candidates = Categorical(logits=candidate_logits)
        options = Categorical(logits=self.option_head(inputs))
        value = self.value_head(inputs).squeeze(-1)
        experts = [Normal(*expert(inputs)) for expert in self.experts]
        return router, candidates, experts, options, value

    def sample(
        self,
        scene_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        latent: torch.Tensor,
        deterministic: bool = False,
    ) -> UniversalSceneAction:
        router, candidates, experts, options, value = self._distributions(
            scene_embedding, candidate_embeddings, candidate_mask, latent
        )
        expert_index = router.probs.argmax(-1) if deterministic else router.sample()
        candidate_index = candidates.probs.argmax(-1) if deterministic else candidates.sample()
        option_index = options.probs.argmax(-1) if deterministic else options.sample()
        means = torch.stack([expert.mean for expert in experts], dim=1)
        scales = torch.stack([expert.stddev for expert in experts], dim=1)
        selected_mean = means[torch.arange(len(expert_index), device=expert_index.device), expert_index]
        selected_scale = scales[torch.arange(len(expert_index), device=expert_index.device), expert_index]
        continuous = Normal(selected_mean, selected_scale)
        raw = continuous.mean if deterministic else continuous.rsample()
        controls = raw.tanh()
        log_prob = (
            router.log_prob(expert_index) + candidates.log_prob(candidate_index) + options.log_prob(option_index)
            + continuous.log_prob(raw).sum(-1) - torch.log(1 - controls.square() + 1e-6).sum(-1)
        )
        return UniversalSceneAction(expert_index, candidate_index, controls, option_index, log_prob, value)

    def evaluate(
        self,
        scene_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        latent: torch.Tensor,
        expert_index: torch.Tensor,
        candidate_index: torch.Tensor,
        controls: torch.Tensor,
        option_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router, candidates, experts, options, value = self._distributions(
            scene_embedding, candidate_embeddings, candidate_mask, latent
        )
        means = torch.stack([expert.mean for expert in experts], dim=1)
        scales = torch.stack([expert.stddev for expert in experts], dim=1)
        continuous = Normal(
            means[torch.arange(len(expert_index), device=expert_index.device), expert_index],
            scales[torch.arange(len(expert_index), device=expert_index.device), expert_index],
        )
        raw = torch.atanh(controls.clamp(-0.999999, 0.999999))
        log_prob = (
            router.log_prob(expert_index) + candidates.log_prob(candidate_index) + options.log_prob(option_index)
            + continuous.log_prob(raw).sum(-1) - torch.log(1 - controls.square() + 1e-6).sum(-1)
        )
        entropy = router.entropy() + candidates.entropy() + options.entropy() + continuous.entropy().sum(-1)
        return log_prob, entropy, value
