"""Small historical-memory and target-evidence conditioned failure ranker."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch import nn

from methods.failure_memory_regression.fm2_features import ParameterSetEncoder, PreLNBlock, parameter_batch


class FM2Model(nn.Module):
    def __init__(self, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.parameters_encoder = ParameterSetEncoder(width, dropout=dropout)
        self.memory_meta = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.role_embedding = nn.Embedding(5, width)
        self.phase_embedding = nn.Embedding(5, width)
        self.witness_query_failure = nn.Parameter(torch.zeros(1, 1, width))
        self.witness_query_pass = nn.Parameter(torch.zeros(1, 1, width))
        self.witness_pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.no_failure_witness = nn.Parameter(torch.zeros(width))
        self.no_pass_witness = nn.Parameter(torch.zeros(width))
        self.memory_set = PreLNBlock(width, 4, dropout)
        self.support_meta = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.support_to_memory = nn.MultiheadAttention(width, 4, batch_first=True)
        self.target_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.gate = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.null_retrieval = nn.Parameter(torch.zeros(width))
        self.no_history = nn.Parameter(torch.zeros(width))
        self.no_support = nn.Parameter(torch.zeros(width))
        self.head = nn.Sequential(nn.Linear(4 * width + 5, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        for token in (self.null_retrieval, self.no_history, self.no_support,
                      self.witness_query_failure, self.witness_query_pass,
                      self.no_failure_witness, self.no_pass_witness):
            nn.init.normal_(token, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(self, values, present, *, spread=None, contrast=None, contrast_mask=None):
        args = parameter_batch(values, present, self.device)
        def convert(item):
            return None if item is None else torch.as_tensor(item, dtype=torch.float32,
                                                               device=self.device)
        return self.parameters_encoder(*args, spread=convert(spread), contrast=convert(contrast),
                                       contrast_mask=convert(contrast_mask))

    def _pool_witnesses(self, values: np.ndarray, mask: np.ndarray,
                        query: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
        count = len(values)
        encoded = self._encode(values.reshape(-1, 4),
                               np.ones((count * 4, 4), dtype=bool)).reshape(count, 4, -1)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=self.device).clone()
        empty = ~valid.any(dim=1)
        valid[empty, 0] = True
        encoded = torch.where(empty[:, None, None] &
                              (torch.arange(4, device=self.device)[None, :, None] == 0),
                              null[None, None], encoded)
        pooled = self.witness_pool(query.expand(count, -1, -1), encoded, encoded,
                                   key_padding_mask=~valid, need_weights=False)[0][:, 0]
        return pooled

    def forward(self, candidate_coords: np.ndarray, memory: dict[str, np.ndarray],
                support: list[dict], *, audit: bool = False):
        n = len(candidate_coords)
        if n == 0:
            raise ValueError("empty candidate batch")
        q = self._encode(candidate_coords, np.ones_like(candidate_coords, dtype=bool))
        k_count = len(memory["values"])
        if k_count:
            geometry = self._encode(memory["values"], memory["present"], spread=memory["spread"],
                                    contrast=memory["contrast"],
                                    contrast_mask=memory["contrast_present"])
            m = geometry + self.memory_meta(torch.as_tensor(memory["metadata"], device=self.device))
            if "failure_witnesses" in memory:
                m = m + self._pool_witnesses(memory["failure_witnesses"], memory["failure_mask"],
                                             self.witness_query_failure, self.no_failure_witness)
                m = m + self._pool_witnesses(memory["pass_witnesses"], memory["pass_mask"],
                                             self.witness_query_pass, self.no_pass_witness)
                m = m + self.role_embedding(torch.as_tensor(memory["role_code"], device=self.device))
                m = m + self.phase_embedding(torch.as_tensor(memory["phase_code"], device=self.device))
            m = self.memory_set(m[None], torch.ones((1, len(m)), dtype=torch.bool,
                                                       device=self.device))[0]
        else:
            m = self.no_history[None]
            geometry = m
        m = torch.cat((m, self.null_retrieval[None]), dim=0)
        retrieval_keys = torch.cat((geometry, self.null_retrieval[None]), dim=0)
        if support:
            s_values = np.stack([row["coords"] for row in support]).astype(np.float32)
            s = self._encode(s_values, np.ones_like(s_values, dtype=bool))
            meta = torch.as_tensor(np.stack([row["features"] for row in support]),
                                   dtype=torch.float32, device=self.device)
            s = s + self.support_meta(meta)
            conditioned = self.support_to_memory(m[None], s[None], s[None],
                                                  need_weights=False)[0][0]
            target_read = self.target_attention(q[None], s[None], s[None],
                                                need_weights=False)[0][0]
            raw_gate = torch.sigmoid(self.gate(torch.cat((q[:, None].expand(-1, len(m), -1),
                                                         m[None].expand(n, -1, -1),
                                                         conditioned[None].expand(n, -1, -1)),
                                                        dim=-1)).squeeze(-1))
            gate = torch.cat((raw_gate[:, :-1], torch.ones((n, 1), device=self.device)), dim=1)
        else:
            target_read = self.no_support[None].expand(n, -1)
            gate = torch.ones((n, len(m)), device=self.device)
        logits = self.query(q) @ self.key(retrieval_keys).T / np.sqrt(q.shape[-1])
        # Source balancing applies to concrete memories; the null token stays available.
        if k_count:
            counts = Counter(memory["source"].tolist())
            balance = torch.as_tensor([np.log(counts[int(src)]) for src in memory["source"]],
                                      dtype=torch.float32, device=self.device)
            logits[:, :k_count] -= balance[None]
        logits = logits + torch.log(gate.clamp_min(1e-6))
        weight = torch.softmax(logits, dim=-1)
        history_read = weight @ self.value(m)
        failure_count = sum(row["label"] == 1 for row in support)
        pass_count = sum(row["label"] == 0 for row in support)
        extra = torch.tensor([np.log1p(failure_count), np.log1p(pass_count),
                              float(weight[:, :-1].mean().detach()) if k_count else 0.0,
                              float(bool(support)), len(support) / 50],
                             dtype=torch.float32, device=self.device)[None].expand(n, -1)
        fused = torch.cat((q, history_read, target_read, history_read - target_read, extra), dim=-1)
        failure_logit = self.head(fused).squeeze(-1)
        if audit:
            return failure_logit, {"attention": weight.detach().cpu().numpy(),
                                   "gate": gate.detach().cpu().numpy()}
        return failure_logit


def support_row(coords: np.ndarray, outcome: dict, history_probability: float) -> dict | None:
    from methods.failure_memory_regression.fm2_schema import valid_label

    label = valid_label(outcome)
    if label is None:
        return None
    ttc = outcome.get("min_ttc")
    clearance = outcome.get("min_clearance")
    role = outcome.get("collision_partner_role")
    phase = outcome.get("observed_maneuver_phase")
    features = np.asarray([
        float(label), float(label - history_probability),
        np.log1p(max(0, float(ttc))) if ttc is not None else 0.0,
        np.sign(float(clearance)) * np.log1p(abs(float(clearance))) if clearance is not None else 0.0,
        float(ttc is not None), float(clearance is not None),
        float(role is not None), float(phase is not None),
    ], dtype=np.float32)
    return {"coords": np.asarray(coords, dtype=np.float32), "label": label,
            "features": features, "history_target_residual": label - history_probability}
