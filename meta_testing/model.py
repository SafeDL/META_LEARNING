"""The single model-level contract used by training and evaluation."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .context.episode_token import EpisodeTokenBuilder
from .context.set_posterior import PosteriorTrainingBatch, SetPosterior, VulnerabilityOutcomeDecoder
from .map.hptr_encoder import HPTRMapEncoder
from .map.schema import MapTokens
from .policy.adversarial_sac import OptionConditionedSAC
from .policy.scene_policy import HybridScenePolicy, SceneAction
from .policy.shared_features import SharedFeatureEncoder
from .scenario.parameter_space import ParameterSpace
from .state import PhysicalStateExtractor
from .training.updates import posterior_elbo


class HierarchicalMetaTester(nn.Module):
    """Compose map, context, Outer and Inner modules without hidden inputs."""

    def __init__(
        self,
        parameter_spaces: Mapping[str, ParameterSpace],
        *,
        state_dim: int,
        map_encoder: HPTRMapEncoder | None = None,
        trajectory_encoder: nn.Module | None = None,
        map_dim: int = 128,
        latent_dim: int = 16,
        token_dim: int = 128,
        outer_history_dim: int = 0,
    ) -> None:
        super().__init__()
        if not parameter_spaces:
            raise ValueError("hierarchical model requires at least one parameter space")
        from .context.trajectory_encoder import TrajectoryEncoder

        self.parameter_spaces = dict(parameter_spaces)
        self.state_dim = int(state_dim)
        if self.state_dim != PhysicalStateExtractor.dimension:
            raise ValueError(f"state_dim must equal the {PhysicalStateExtractor.dimension}-D physical Inner state")
        self.map_encoder = map_encoder or HPTRMapEncoder(embedding_dim=map_dim)
        if self.map_encoder.embedding_dim != map_dim:
            raise ValueError("map encoder embedding dimension must equal map_dim")
        trajectory = trajectory_encoder or TrajectoryEncoder(hidden_dim=map_dim)
        self.episode_token_builder = EpisodeTokenBuilder(map_dim, max(space.continuous_dim for space in parameter_spaces.values()), len(next(iter(parameter_spaces.values())).options), trajectory, token_dim)
        self.posterior = SetPosterior(token_dim, latent_dim)
        self.outcome_decoder = VulnerabilityOutcomeDecoder(latent_dim, map_dim, max(space.continuous_dim for space in parameter_spaces.values()), len(next(iter(parameter_spaces.values())).options))
        self.outer_history_dim = int(outer_history_dim)
        self.scene_policies = nn.ModuleDict({
            identifier: HybridScenePolicy(map_dim + latent_dim + self.outer_history_dim, len(space.candidates), space.continuous_dim, len(space.options))
            for identifier, space in parameter_spaces.items()
        })
        self.option_embedding = nn.Embedding(len(next(iter(parameter_spaces.values())).options), 16)
        self.shared_feature_encoder = SharedFeatureEncoder(self.state_dim, map_dim, latent_dim, 16, max(space.continuous_dim for space in parameter_spaces.values()))
        self.inner_sac = OptionConditionedSAC(256)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_map(self, tokens: MapTokens) -> tuple[torch.Tensor, torch.Tensor]:
        return self.map_encoder(tokens)

    def infer_posterior(self, support_tokens: torch.Tensor, support_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.posterior(support_tokens, support_mask)

    def posterior_loss(self, batch: PosteriorTrainingBatch) -> torch.Tensor:
        batch.validate()
        mean, logvar = self.infer_posterior(batch.support_tokens, batch.support_mask)
        logits = self.outcome_decoder(mean, batch.target_map, batch.target_config, batch.target_option)
        return posterior_elbo(logits, batch.target_outcome, mean, logvar)

    def select_scene(self, parameter_space_id: str, map_embedding: torch.Tensor, latent: torch.Tensor, history: torch.Tensor | None = None, *, deterministic: bool = False) -> SceneAction:
        if parameter_space_id not in self.scene_policies:
            raise ValueError(f"unknown parameter space {parameter_space_id!r}")
        if map_embedding.ndim == 1:
            map_embedding, latent = map_embedding.unsqueeze(0), latent.unsqueeze(0)
        if history is None:
            history = map_embedding.new_zeros((map_embedding.shape[0], self.outer_history_dim))
        if history.shape != (map_embedding.shape[0], self.outer_history_dim):
            raise ValueError("outer history shape does not match model configuration")
        return self.scene_policies[parameter_space_id].sample(torch.cat((map_embedding, latent, history), dim=-1), deterministic)

    def inner_features(self, state: torch.Tensor, map_embedding: torch.Tensor, latent: torch.Tensor, option_index: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        return self.shared_feature_encoder(state, map_embedding, latent, self.option_embedding(option_index), config)

    def act_inner(self, state: torch.Tensor, map_embedding: torch.Tensor, latent: torch.Tensor, option_index: torch.Tensor, config: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        return self.inner_sac.act(self.inner_features(state, map_embedding, latent, option_index, config), deterministic)

    def training_components(self) -> dict[str, nn.Module]:
        return {
            "map_encoder": self.map_encoder,
            "episode_token_builder": self.episode_token_builder,
            "posterior": self.posterior,
            "outcome_decoder": self.outcome_decoder,
            "scene_policies": self.scene_policies,
            "option_embedding": self.option_embedding,
            "shared_feature_encoder": self.shared_feature_encoder,
            "inner_sac": self.inner_sac,
        }
