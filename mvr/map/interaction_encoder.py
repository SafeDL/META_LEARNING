"""Encode route-pair conflict candidates on top of HPTR lane embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ..scenario.interaction import InteractionCandidate
from .schema import MapTokens


@dataclass(frozen=True)
class SceneEncoding:
    global_embedding: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor


class InteractionEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.descriptor = nn.Sequential(
            nn.Linear(6, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.combine = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.global_combine = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )

    @staticmethod
    def _lane_lookup(tokens: MapTokens) -> dict[tuple[object, object, int], int]:
        return {tuple(polyline.attributes["lane_index"]): index for index, polyline in enumerate(tokens.polylines)}

    def forward(
        self,
        local_lanes: torch.Tensor,
        map_embedding: torch.Tensor,
        tokens: MapTokens,
        candidates: Sequence[InteractionCandidate],
    ) -> SceneEncoding:
        if not candidates:
            raise ValueError("scene encoding requires at least one interaction candidate")
        lookup = self._lane_lookup(tokens)
        encoded = []
        for candidate in candidates:
            try:
                sut = local_lanes[torch.as_tensor([lookup[tuple(lane)] for lane in candidate.sut_route], device=local_lanes.device)].mean(0)
                adversary = local_lanes[torch.as_tensor([lookup[tuple(lane)] for lane in candidate.adversary_route], device=local_lanes.device)].mean(0)
            except KeyError as error:
                raise ValueError("interaction route is absent from map tokens") from error
            descriptor = torch.as_tensor(candidate.features(), device=local_lanes.device)
            encoded.append(self.combine(torch.cat((sut, adversary, self.descriptor(descriptor)), dim=-1)))
        candidate_embeddings = torch.stack(encoded)
        scene = self.global_combine(torch.cat((map_embedding, candidate_embeddings.mean(0)), dim=-1))
        return SceneEncoding(scene, candidate_embeddings, torch.ones(len(candidates), dtype=torch.bool, device=scene.device))
