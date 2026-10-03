"""Source-separated full-context historical SRD-TNP m/h predictor."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import HistoryOutput
from .tokens import ScenarioTokens
from .krblock import KRBlock


class HistoricalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = ScenarioTokens(config["d_model"], config["fourier_features"], config["fourier_scales"])
        self.blocks = nn.ModuleList([KRBlock(config["d_model"], config["heads"], config["ffn_dim"],
            config["dropout"], config["query_chunk"], config["key_chunk"]) for _ in range(config["layers"])])
        self.mean_head = nn.Linear(config["d_model"], 1)
        self.source_gate = nn.Sequential(nn.Linear(config["d_model"], 32), nn.GELU(), nn.Linear(32, 1))
        self.h_head = nn.Linear(config["d_model"], config["h_dim"])

    def as_tensor(self, value):
        parameter = next(self.parameters())
        return torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)

    def forward(self, contexts, x):
        x = self.as_tensor(x)
        query_token = self.tokens(x)
        representations = []
        for context in contexts:
            mask = np.asarray(context.valid, bool)
            if not mask.any():
                continue
            c = self.tokens(self.as_tensor(context.x[mask]), self.as_tensor(context.z[mask]))
            q = query_token
            for block in self.blocks:
                c, q = block(c, q)
            representations.append(q)
        if representations:
            per_source = torch.stack(representations, dim=1)
            weights = torch.softmax(self.source_gate(per_source), dim=1)
            source_mean = self.mean_head(per_source)
            m = (weights * source_mean).sum(1)
            fused = (weights * per_source).sum(1)
        else:
            fused = query_token
            m = self.mean_head(fused)
        h = F.normalize(self.h_head(fused), p=2, dim=-1, eps=1e-12)
        return m, h

    @torch.no_grad()
    def predict(self, contexts, x, batch_size=128):
        if self.training:
            raise ValueError("predict requires eval/frozen model")
        means, embeddings = [], []
        cache = self.encode_contexts(contexts)
        self.deployment_context_cache = cache
        for i in range(0, len(x), batch_size):
            m, h = self.predict_with_context_cache(cache, x[i:i + batch_size])
            means.append(m.cpu().numpy())
            embeddings.append(h.cpu().numpy())
        return HistoryOutput(np.concatenate(means), np.concatenate(embeddings))

    @torch.no_grad()
    def encode_contexts(self, contexts):
        if self.training:
            raise ValueError("context cache requires eval mode")
        cache = []
        for source in contexts:
            valid = np.asarray(source.valid, bool)
            if not valid.any():
                continue
            c = self.tokens(self.as_tensor(source.x[valid]), self.as_tensor(source.z[valid]))
            layers = []
            for block in self.blocks:
                layers.append(c)
                c = block.update(c, c)
            cache.append(layers)
        return cache

    @torch.no_grad()
    def predict_with_context_cache(self, cache, x):
        query_token = self.tokens(self.as_tensor(x))
        outputs = []
        for source_layers in cache:
            q = query_token
            for block, context in zip(self.blocks, source_layers):
                q = block.update(q, context)
            outputs.append(q)
        if outputs:
            per_source = torch.stack(outputs, dim=1)
            weights = torch.softmax(self.source_gate(per_source), dim=1)
            m = (weights * self.mean_head(per_source)).sum(1)
            fused = (weights * per_source).sum(1)
        else:
            fused = query_token
            m = self.mean_head(fused)
        return m, F.normalize(self.h_head(fused), p=2, dim=-1, eps=1e-12)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self
