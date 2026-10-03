"""Shared context/query Pre-LN kernel-regression transformer block."""
from torch import nn

from .scan_attention import scan_attention


class KRBlock(nn.Module):
    def __init__(self, d_model=128, heads=4, ffn_dim=256, dropout=.1, query_chunk=128, key_chunk=256):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must divide heads")
        self.heads = heads
        self.query_chunk, self.key_chunk = query_chunk, key_chunk
        self.ln_attention = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def heads_of(self, value):
        return value.reshape(len(value), self.heads, -1).transpose(0, 1).unsqueeze(0)

    def attend(self, query, context):
        q = self.heads_of(self.q_proj(self.ln_attention(query)))
        c = self.ln_attention(context)
        k, v = self.heads_of(self.k_proj(c)), self.heads_of(self.v_proj(c))
        out = scan_attention(q, k, v, self.query_chunk, self.key_chunk)
        return self.out_proj(out.squeeze(0).transpose(0, 1).reshape(len(query), -1))

    def update(self, query, context):
        value = query + self.dropout(self.attend(query, context))
        return value + self.dropout(self.ffn(self.ln_ffn(value)))

    def forward(self, context, query):
        # Both paths see the old context of this block and share all weights.
        return self.update(context, context), self.update(query, context)
