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
    def __init__(self, scene_dim: int, latent_dim: int, continuous_dim: int, num_experts: int) -> None:
        super().__init__()
        inputs = scene_dim + latent_dim
        self.router = TaskAwareMoERouter(scene_dim, latent_dim, num_experts)
        self.experts = nn.ModuleList(_InitialConditionExpert(inputs, continuous_dim) for _ in range(num_experts))
        self.candidate_scorer = nn.Sequential(nn.Linear(2 * scene_dim + latent_dim, 256), nn.ReLU(), nn.Linear(256, 1))
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
    ) -> tuple[Categorical, Categorical, list[Normal], torch.Tensor]:
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
        value = self.value_head(inputs).squeeze(-1)
        experts = [Normal(*expert(inputs)) for expert in self.experts]
        return router, candidates, experts, value

    def _continuous_mask(
        self, mask: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch_size, self.continuous_dim), dtype=torch.bool, device=device)
        value = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape == (1, self.continuous_dim):
            value = value.expand(batch_size, -1)
        if value.shape != (batch_size, self.continuous_dim):
            raise ValueError("continuous action mask must align with the policy batch")
        return value

    def _continuous_bounds(
        self, bounds: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bounds is None:
            value = torch.tensor([[-1.0, 1.0]] * self.continuous_dim, device=device)
        else:
            value = torch.as_tensor(bounds, dtype=torch.float32, device=device)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.shape[0] == 1:
            value = value.expand(batch_size, -1, -1)
        if value.shape != (batch_size, self.continuous_dim, 2):
            raise ValueError("continuous action bounds must align with the policy batch")
        lower, upper = value[..., 0], value[..., 1]
        if not torch.all(lower < upper):
            raise ValueError("continuous action bounds must be ordered")
        return lower, upper

    def sample(
        self,
        scene_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        latent: torch.Tensor,
        deterministic: bool = False,
        continuous_mask: torch.Tensor | None = None,
        continuous_bounds: torch.Tensor | None = None,
    ) -> UniversalSceneAction:
        router, candidates, experts, value = self._distributions(
            scene_embedding, candidate_embeddings, candidate_mask, latent
        )
        expert_index = router.probs.argmax(-1) if deterministic else router.sample()
        candidate_index = candidates.probs.argmax(-1) if deterministic else candidates.sample()
        means = torch.stack([expert.mean for expert in experts], dim=1)
        scales = torch.stack([expert.stddev for expert in experts], dim=1)
        selected_mean = means[torch.arange(len(expert_index), device=expert_index.device), expert_index]
        selected_scale = scales[torch.arange(len(expert_index), device=expert_index.device), expert_index]
        continuous = Normal(selected_mean, selected_scale)
        raw = continuous.mean if deterministic else continuous.rsample()
        mask = self._continuous_mask(continuous_mask, len(expert_index), raw.device)
        lower, upper = self._continuous_bounds(continuous_bounds, len(expert_index), raw.device)
        unit = raw.tanh()
        controls = (lower + 0.5 * (unit + 1.0) * (upper - lower)) * mask
        log_prob = (
            router.log_prob(expert_index) + candidates.log_prob(candidate_index)
            + (continuous.log_prob(raw) * mask).sum(-1)
            - (torch.log(1 - unit.square() + 1e-6) * mask).sum(-1)
            - (torch.log(0.5 * (upper - lower)) * mask).sum(-1)
        )
        return UniversalSceneAction(expert_index, candidate_index, controls, log_prob, value)

    def evaluate(
        self,
        scene_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        latent: torch.Tensor,
        expert_index: torch.Tensor,
        candidate_index: torch.Tensor,
        controls: torch.Tensor,
        continuous_mask: torch.Tensor | None = None,
        continuous_bounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router, candidates, experts, value = self._distributions(
            scene_embedding, candidate_embeddings, candidate_mask, latent
        )
        means = torch.stack([expert.mean for expert in experts], dim=1)
        scales = torch.stack([expert.stddev for expert in experts], dim=1)
        continuous = Normal(
            means[torch.arange(len(expert_index), device=expert_index.device), expert_index],
            scales[torch.arange(len(expert_index), device=expert_index.device), expert_index],
        )
        mask = self._continuous_mask(continuous_mask, len(expert_index), controls.device)
        lower, upper = self._continuous_bounds(continuous_bounds, len(expert_index), controls.device)
        unit = 2.0 * (controls - lower) / (upper - lower) - 1.0
        raw = torch.atanh(unit.clamp(-0.999999, 0.999999))
        log_prob = (
            router.log_prob(expert_index) + candidates.log_prob(candidate_index)
            + (continuous.log_prob(raw) * mask).sum(-1)
            - (torch.log(1 - unit.square() + 1e-6) * mask).sum(-1)
            - (torch.log(0.5 * (upper - lower)) * mask).sum(-1)
        )
        entropy = (
            router.entropy() + candidates.entropy()
            + (continuous.entropy() * mask).sum(-1)
            + (torch.log(0.5 * (upper - lower)) * mask).sum(-1)
        )
        return log_prob, entropy, value
