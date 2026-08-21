"""Compact HPTR-style heterogeneous polyline encoder (map encoder only)."""
from __future__ import annotations

import math
import torch
from torch import nn

from .relations import RELATION_TYPES
from .schema import MapTokens


class _RelativeAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dim must divide evenly across heads")
        self.heads, self.head_dim = heads, dim // heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.pose_bias = nn.Sequential(nn.Linear(3, dim), nn.Tanh(), nn.Linear(dim, heads))
        self.relation_bias = nn.ParameterDict({relation: nn.Parameter(torch.zeros(heads)) for relation in RELATION_TYPES})
        self.output = nn.Linear(dim, dim)

    def forward(self, features: torch.Tensor, centres: torch.Tensor, headings: torch.Tensor, relations: dict[str, tuple[tuple[int, int], ...]]) -> torch.Tensor:
        count, dim = features.shape
        q = self.query(features).view(count, self.heads, self.head_dim).transpose(0, 1)
        k = self.key(features).view(count, self.heads, self.head_dim).transpose(0, 1)
        v = self.value(features).view(count, self.heads, self.head_dim).transpose(0, 1)
        relative = centres[None, :, :] - centres[:, None, :]
        c, s = torch.cos(headings)[:, None], torch.sin(headings)[:, None]
        local_x = c * relative[..., 0] + s * relative[..., 1]
        local_y = -s * relative[..., 0] + c * relative[..., 1]
        delta_heading = torch.atan2(torch.sin(headings[None, :] - headings[:, None]), torch.cos(headings[None, :] - headings[:, None]))
        pose = torch.stack((local_x, local_y, delta_heading), dim=-1)
        score = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
        score = score + self.pose_bias(pose).permute(2, 0, 1)
        for relation, edges in relations.items():
            if relation not in self.relation_bias:
                continue
            for source, target in edges:
                score[:, source, target] = score[:, source, target] + self.relation_bias[relation]
        attention = score.softmax(dim=-1)
        attended = torch.einsum("hnm,hmd->hnd", attention, v).transpose(0, 1).reshape(count, dim)
        return self.output(attended)


class HPTRMapEncoder(nn.Module):
    """SE(2)-aware map encoder yielding local lane tokens and global map state."""
    def __init__(self, embedding_dim: int = 128, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        # local XY, local heading cos/sin, curvature, lane width, speed limit
        self.point_encoder = nn.Sequential(nn.Linear(7, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.layers = nn.ModuleList(nn.ModuleDict({"attention": _RelativeAttention(embedding_dim, heads), "norm1": nn.LayerNorm(embedding_dim), "ff": nn.Sequential(nn.Linear(embedding_dim, 2 * embedding_dim), nn.ReLU(), nn.Linear(2 * embedding_dim, embedding_dim)), "norm2": nn.LayerNorm(embedding_dim)}) for _ in range(layers))
        self.pool_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, 1))

    def forward(self, tokens: MapTokens) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        points, centres, headings = tokens.tensorize(device)
        local = self.point_encoder(points).mean(dim=1)
        for layer in self.layers:
            local = layer["norm1"](local + layer["attention"](local, centres, headings, dict(tokens.relations)))
            local = layer["norm2"](local + layer["ff"](local))
        weights = self.pool_gate(local).squeeze(-1).softmax(dim=0)
        return local, torch.sum(weights[:, None] * local, dim=0)
