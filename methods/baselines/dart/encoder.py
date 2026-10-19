"""Source-separated historical features; each query attends only to context."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.baselines.hgt.config import group_of

from .config import (
    CONTEXT_PER_FAMILY,
    FEATURES,
    HEADS,
    INITIAL_AMPLITUDE,
    MAX_AMPLITUDE,
    WIDTH,
)


def fixed_context(history, seed):
    contexts = {}
    for position, (name, source) in enumerate(history.items()):
        generator = np.random.default_rng(940001 + seed + position)
        indices = np.concatenate(
            [
                generator.choice(
                    np.flatnonzero(source["x"][:, 4] == family),
                    CONTEXT_PER_FAMILY,
                    replace=False,
                )
                for family in np.unique(source["x"][:, 4])
            ]
        )
        contexts[name] = {"x": source["x"][indices], "risk": source["risk"][indices]}
    return contexts


def group_weights(names):
    groups = [group_of(name) for name in names]
    return np.asarray([1 / (len(set(groups)) * groups.count(g)) for g in groups])


class ResponseSetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_input = nn.Linear(6, WIDTH)
        self.query_input = nn.Linear(5, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH,
            HEADS,
            2 * WIDTH,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer, 2, enable_nested_tensor=False
        )
        self.cross_attention = nn.MultiheadAttention(
            WIDTH, HEADS, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(WIDTH),
            nn.Linear(WIDTH, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, FEATURES),
        )
        fraction = INITIAL_AMPLITUDE / MAX_AMPLITUDE
        self.raw_amplitude = nn.Parameter(
            torch.tensor(math.log(fraction / (1 - fraction)))
        )

    @property
    def amplitude(self):
        return MAX_AMPLITUDE * self.raw_amplitude.sigmoid()

    def features(self, x, contexts):
        parameter = next(self.parameters())
        x = torch.as_tensor(x, device=parameter.device, dtype=parameter.dtype)
        source_features = []
        for context in contexts.values():
            cx = torch.as_tensor(context["x"], device=x.device, dtype=x.dtype)
            cr = torch.as_tensor(context["risk"], device=x.device, dtype=x.dtype)
            feature = x.new_zeros((len(x), FEATURES))
            for family in (0, 1):
                rows = torch.nonzero(x[:, 4] == family).flatten()
                if not len(rows):
                    continue
                tokens = torch.cat(
                    (cx[cx[:, 4] == family], cr[cx[:, 4] == family, None]), 1
                )
                context_tokens = self.context_encoder(self.context_input(tokens)[None])
                queries = self.query_input(x[rows])[None]
                attended, _ = self.cross_attention(
                    queries, context_tokens, context_tokens, need_weights=False
                )
                feature[rows] = F.normalize(
                    self.output((queries + attended)[0]), dim=-1
                )
            source_features.append(feature)
        return torch.stack(source_features)

    def covariance(self, x, contexts, base_covariance):
        feature = self.features(x, contexts)
        weights = torch.as_tensor(
            group_weights(tuple(contexts)), device=feature.device, dtype=feature.dtype
        )
        residual = torch.einsum("s,sid,sjd->ij", weights, feature, feature)
        inputs = torch.as_tensor(x, device=feature.device)
        residual = residual * (inputs[:, 4:5] == inputs[:, 4:5].T)
        base = torch.as_tensor(
            base_covariance, device=feature.device, dtype=torch.float64
        )
        return base + self.amplitude.double().square() * residual.double()
