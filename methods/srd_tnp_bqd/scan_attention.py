"""Exact online-softmax scan with tiled backward, plus fused CUDA execution.

Only Q,K,V, output and row log-normalizers are saved, never Nq-by-Nk scores.
The CUDA fused path has the same complete-key normalization and a supported
memory-efficient backward. No context subsampling occurs in either path.
"""
import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, query_chunk, key_chunk):
        scale = 1 / math.sqrt(q.shape[-1])
        out = torch.zeros_like(q)
        lognorm = q.new_full((*q.shape[:-1], 1), -torch.inf)
        for i in range(0, q.shape[-2], query_chunk):
            qi = q[..., i:i + query_chunk, :]
            maximum = qi.new_full((*qi.shape[:-1], 1), -torch.inf)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(qi)
            for j in range(0, k.shape[-2], key_chunk):
                scores = (qi @ k[..., j:j + key_chunk, :].transpose(-1, -2)) * scale
                next_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                rescale = torch.exp(maximum - next_max)
                weights = torch.exp(scores - next_max)
                denominator = rescale * denominator + weights.sum(-1, keepdim=True)
                numerator = rescale * numerator + weights @ v[..., j:j + key_chunk, :]
                maximum = next_max
            if k.shape[-2]:
                out[..., i:i + query_chunk, :] = numerator / denominator
                lognorm[..., i:i + query_chunk, :] = maximum + denominator.log()
        ctx.save_for_backward(q, k, v, out, lognorm)
        ctx.query_chunk, ctx.key_chunk = query_chunk, key_chunk
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lognorm = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        scale = 1 / math.sqrt(q.shape[-1])
        for i in range(0, q.shape[-2], ctx.query_chunk):
            sl = slice(i, i + ctx.query_chunk)
            qi, go = q[..., sl, :], grad[..., sl, :]
            correction = (go * out[..., sl, :]).sum(-1, keepdim=True)
            for j in range(0, k.shape[-2], ctx.key_chunk):
                sk = slice(j, j + ctx.key_chunk)
                kj, vj = k[..., sk, :], v[..., sk, :]
                prob = torch.exp((qi @ kj.transpose(-1, -2)) * scale - lognorm[..., sl, :])
                dv[..., sk, :] += prob.transpose(-1, -2) @ go
                ds = prob * (go @ vj.transpose(-1, -2) - correction)
                dq[..., sl, :] += (ds @ kj) * scale
                dk[..., sk, :] += (ds.transpose(-1, -2) @ qi) * scale
        return dq, dk, dv, None, None


def scan_attention(q, k, v, query_chunk=128, key_chunk=256, fused=True):
    if query_chunk <= 0 or key_chunk <= 0:
        raise ValueError("positive attention chunk sizes required")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2] or q.shape[-1] != v.shape[-1]:
        raise ValueError("invalid attention shapes")
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16, torch.float32) and fused and k.shape[-2]:
        # A fused global softmax is an allowed supported implementation of the
        # scan. Explicitly exclude the quadratic-memory MATH fallback.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return torch.cat([F.scaled_dot_product_attention(q[..., i:i + query_chunk, :].contiguous(),
                k.contiguous(), v.contiguous(), dropout_p=0.0) for i in range(0, q.shape[-2], query_chunk)], dim=-2)
    return _Scan.apply(q, k, v, query_chunk, key_chunk)


def dense_attention(q, k, v):
    return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1) @ v
