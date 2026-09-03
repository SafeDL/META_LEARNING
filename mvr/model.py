"""The transferable map- and interaction-conditioned scenario miner."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .context.episode_token import EpisodeTokenBuilder
from .context.outcome_decoder import PosteriorTrainingBatch, VulnerabilityOutcomeDecoder
from .context.pearl_context import PearlContextEncoder
from .context.trajectory_encoder import TrajectoryEncoder
from .map.hptr_encoder import HPTRMapEncoder
from .map.interaction_encoder import InteractionEncoder, SceneEncoding
from .map.schema import MapTokens
from .policy.adversarial_sac import AdversarialSAC
from .policy.shared_features import SharedFeatureEncoder
from .policy.universal_scene_policy import UniversalSceneAction, UniversalScenePolicy
from .scenario.interaction import InteractionCandidate
from .state import PhysicalStateExtractor
from .training.updates import posterior_elbo


class TransferableScenarioMiner(nn.Module):
    """Shared geometry-aware Outer policy, context inference, and Inner SAC."""

    def __init__(
        self,
        *,
        state_dim: int,
        map_encoder: HPTRMapEncoder | None = None,
        map_dim: int = 128,
        latent_dim: int = 16,
        token_dim: int = 128,
        continuous_dim: int = 5,
        inner_action_dim: int = 4,
        num_experts: int = 4,
        context_kl_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        if state_dim != PhysicalStateExtractor.dimension:
            raise ValueError(f"state_dim must equal the {PhysicalStateExtractor.dimension}-D physical Inner state")
        self.state_dim = state_dim
        self.context_kl_weight = float(context_kl_weight)
        self.map_encoder = map_encoder or HPTRMapEncoder(embedding_dim=map_dim)
        if self.map_encoder.embedding_dim != map_dim:
            raise ValueError("map encoder embedding dimension must equal map_dim")
        self.interaction_encoder = InteractionEncoder(map_dim)
        self.continuous_dim = continuous_dim
        self.task_structure_encoder = nn.Sequential(
            nn.Linear(3 * continuous_dim, map_dim), nn.ReLU(), nn.Linear(map_dim, map_dim)
        )
        trajectory = TrajectoryEncoder(hidden_dim=map_dim)
        self.concrete_dim = map_dim + 2 * continuous_dim
        self.episode_token_builder = EpisodeTokenBuilder(
            map_dim, self.concrete_dim, trajectory, token_dim
        )
        self.context_encoder = PearlContextEncoder(token_dim, latent_dim)
        self.outcome_decoder = VulnerabilityOutcomeDecoder(
            latent_dim, map_dim, self.concrete_dim
        )
        self.universal_scene_policy = UniversalScenePolicy(
            map_dim, latent_dim, continuous_dim, num_experts
        )
        self.shared_feature_encoder = SharedFeatureEncoder(
            self.state_dim, map_dim, latent_dim, self.concrete_dim
        )
        self.inner_sac = AdversarialSAC(
            256, action_dim=inner_action_dim, context_dim=latent_dim
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_map(self, tokens: MapTokens) -> tuple[torch.Tensor, torch.Tensor]:
        return self.map_encoder(tokens)

    def encode_scene(
        self, tokens: MapTokens, candidates: Sequence[InteractionCandidate]
    ) -> SceneEncoding:
        local, global_embedding = self.encode_map(tokens)
        return self.interaction_encoder(local, global_embedding, tokens, candidates)

    def encode_task_structure(
        self,
        scene_embedding: torch.Tensor,
        logical_domain_bounds: dict[str, object],
        logical_parameter_mask: Sequence[bool],
    ) -> torch.Tensor:
        """Combine map/interaction structure with observable Logical bounds."""
        mask = [bool(value) for value in logical_parameter_mask]
        values = [
            float(value) * float(active)
            for name, active in zip(logical_domain_bounds, mask)
            for value in logical_domain_bounds[name]
        ]
        if len(values) != 2 * self.continuous_dim or len(mask) != self.continuous_dim:
            raise ValueError("task structure requires padded Logical bounds and a matching mask")
        domain = torch.tensor(
            values + [float(value) for value in mask],
            dtype=scene_embedding.dtype,
            device=scene_embedding.device,
        )
        return scene_embedding + self.task_structure_encoder(domain)

    def concrete_features(
        self,
        candidate_embedding: torch.Tensor,
        continuous: torch.Tensor,
        logical_parameter_mask: Sequence[bool] | torch.Tensor,
    ) -> torch.Tensor:
        """Build padded concrete input without leaking inactive Logical values."""
        if candidate_embedding.ndim == 1:
            candidate_embedding = candidate_embedding.unsqueeze(0)
        if continuous.ndim == 1:
            continuous = continuous.unsqueeze(0)
        mask = torch.as_tensor(
            logical_parameter_mask, dtype=continuous.dtype, device=continuous.device
        )
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1:
            mask = mask.expand(continuous.shape[0], -1)
        if continuous.shape[-1] != self.continuous_dim or mask.shape != continuous.shape:
            raise ValueError("concrete Logical controls must match the model continuous dimension")
        return torch.cat((candidate_embedding, continuous * mask, mask), dim=-1)

    def infer_posterior(
        self, support_tokens: torch.Tensor, support_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(support_tokens, support_mask)

    def posterior_loss(self, batch: PosteriorTrainingBatch) -> torch.Tensor:
        batch.validate()
        mean, logvar = self.infer_posterior(batch.support_tokens, batch.support_mask)
        latent = self.context_encoder.sample(mean, logvar)
        logits = self.outcome_decoder(latent, batch.target_scene, batch.target_concrete)
        return posterior_elbo(
            logits, batch.target_outcome, mean, logvar, kl_weight=self.context_kl_weight
        )

    def select_scene(
        self,
        encoding: SceneEncoding,
        latent: torch.Tensor,
        *,
        deterministic: bool = False,
        continuous_mask: torch.Tensor | None = None,
        continuous_bounds: torch.Tensor | None = None,
    ) -> UniversalSceneAction:
        return self.universal_scene_policy.sample(
            encoding.global_embedding, encoding.candidate_embeddings,
            encoding.candidate_mask, latent, deterministic, continuous_mask, continuous_bounds,
        )

    def inner_features(
        self,
        state: torch.Tensor,
        scene_embedding: torch.Tensor,
        latent: torch.Tensor,
        concrete: torch.Tensor,
    ) -> torch.Tensor:
        return self.shared_feature_encoder(
            state, scene_embedding, latent, concrete
        )

    def act_inner(
        self,
        state: torch.Tensor,
        scene_embedding: torch.Tensor,
        latent: torch.Tensor,
        concrete: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.inner_sac.act(
            self.inner_features(state, scene_embedding, latent, concrete),
            deterministic,
            context=latent,
        )

    def training_components(self) -> dict[str, nn.Module]:
        return {
            "map_encoder": self.map_encoder,
            "interaction_encoder": self.interaction_encoder,
            "task_structure_encoder": self.task_structure_encoder,
            "episode_token_builder": self.episode_token_builder,
            "context_encoder": self.context_encoder,
            "outcome_decoder": self.outcome_decoder,
            "universal_scene_policy": self.universal_scene_policy,
            "shared_feature_encoder": self.shared_feature_encoder,
            "inner_sac": self.inner_sac,
        }
