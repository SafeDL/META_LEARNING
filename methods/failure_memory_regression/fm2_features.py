"""Schema-aware, order-invariant numeric parameter set encoding."""

from __future__ import annotations

import torch
from torch import nn


class PreLNBlock(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width * 2, width))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        x = x + self.attn(z, z, z, key_padding_mask=~mask, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x * mask.unsqueeze(-1)


class ParameterSetEncoder(nn.Module):
    """No position embedding; semantic IDs travel with their numeric values."""

    def __init__(self, width: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.numeric = nn.Linear(21, width)  # 16 PLE + u + outside + spread + contrast + mask
        self.name = nn.Embedding(5, width)
        self.role = nn.Embedding(5, width)
        self.group = nn.Embedding(6, width)
        self.kind = nn.Embedding(4, width)
        self.blocks = nn.ModuleList(PreLNBlock(width, heads, dropout) for _ in range(layers))
        self.pool_query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor, present: torch.Tensor,
                name: torch.Tensor, role: torch.Tensor, group: torch.Tensor,
                kind: torch.Tensor, *, spread: torch.Tensor | None = None,
                contrast: torch.Tensor | None = None,
                contrast_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Shape: [batch, number of parameters]. Present masks missing axes and padding.
        if spread is None:
            spread = torch.zeros_like(value)
        if contrast is None:
            contrast = torch.zeros_like(value)
        if contrast_mask is None:
            contrast_mask = torch.zeros_like(value)
        boundaries = torch.arange(16, device=value.device, dtype=value.dtype) / 16
        ple = ((value.unsqueeze(-1) - boundaries) * 16).clamp(0, 1)
        numeric = torch.cat((ple, value.unsqueeze(-1),
                             ((value < 0) | (value > 1)).float().unsqueeze(-1),
                             spread.unsqueeze(-1), contrast.unsqueeze(-1),
                             contrast_mask.unsqueeze(-1)), dim=-1)
        x = (self.numeric(numeric) + self.name(name) + self.role(role) +
             self.group(group) + self.kind(kind))
        safe = present.bool().clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
            x[empty, 0] = 0
        x = x * safe.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, safe)
        pooled = self.pool(self.pool_query.expand(len(x), -1, -1), x, x,
                           key_padding_mask=~safe, need_weights=False)[0][:, 0]
        return self.output_norm(pooled)


def parameter_batch(values, present, device: torch.device):
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    mask = torch.as_tensor(present, dtype=torch.bool, device=device)
    if x.ndim == 1:
        x, mask = x[None], mask[None]
    batch, length = x.shape
    # The S01 wrapper supplies the four declared semantic IDs; the encoder itself
    # accepts any length and reserves ID 0 for an unrecognized future field.
    ids = torch.tensor(([1, 2, 3, 4] + [0] * max(0, length - 4))[:length],
                       device=device)[None].expand(batch, -1)
    role = torch.ones_like(ids)
    group = torch.tensor(([1, 2, 3, 3] + [0] * max(0, length - 4))[:length],
                         device=device)[None].expand(batch, -1)
    kind = torch.ones_like(ids)
    return x, mask, ids, role, group, kind
