"""Small native-PyTorch GAT SEM with node, line-graph edge and global branches."""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        if out_features % heads:
            raise ValueError("out_features must be divisible by heads")
        self.heads = heads
        self.head_dim = out_features // heads
        self.query = nn.Linear(in_features, out_features, bias=False)
        self.key = nn.Linear(in_features, out_features, bias=False)
        self.value = nn.Linear(in_features, out_features, bias=False)
        self.output = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, entities, _ = features.shape
        shape = (batch, entities, self.heads, self.head_dim)
        q = self.query(features).view(shape).transpose(1, 2)
        k = self.key(features).view(shape).transpose(1, 2)
        v = self.value(features).view(shape).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = adjacency.unsqueeze(1) > 0
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        values = torch.matmul(attention, v).transpose(1, 2).reshape(batch, entities, -1)
        return self.output(values)


class GraphBranch(nn.Module):
    def __init__(self, in_features: int, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.gat1 = MultiHeadGraphAttention(in_features, hidden, heads, dropout)
        self.gat2 = MultiHeadGraphAttention(hidden, hidden, heads, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.norm1(self.gat1(x, adjacency))))
        x = self.dropout(torch.relu(self.norm2(self.gat2(x, adjacency))))
        return x.mean(dim=1)


class ScenarioEvaluationModel(nn.Module):
    """Binary classifier returning logits; sigmoid is prediction-only."""

    def __init__(self, node_dim: int = 9, edge_dim: int = 6, global_dim: int = 8, hidden: int = 64, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_branch = GraphBranch(node_dim, hidden, heads, dropout)
        self.edge_branch = GraphBranch(edge_dim, hidden, heads, dropout)
        self.global_branch = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node = self.node_branch(batch["nodes"], batch["node_adj"])
        edge = self.edge_branch(batch["edges"], batch["line_adj"])
        glob = self.global_branch(batch["global_features"])
        return torch.cat((node, edge, glob), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.embedding(batch)).squeeze(-1)

