"""Small dependency-free ranking metrics for the MVP reports."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(predicted: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k using non-negative continuous vulnerability relevance."""
    if k < 1:
        raise ValueError("k must be positive")
    k = min(k, len(truth))
    order = np.argsort(-predicted, kind="stable")[:k]
    ideal = np.argsort(-truth, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((2.0**truth[order] - 1.0) * discounts))
    idcg = float(np.sum((2.0**truth[ideal] - 1.0) * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def top_k_recall(predicted: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """Fraction of true top-k anchors recovered by a prediction top-k."""
    predicted_top = set(np.argsort(-predicted, kind="stable")[:k])
    truth_top = set(np.argsort(-truth, kind="stable")[:k])
    return len(predicted_top & truth_top) / min(k, len(truth))


def spearman_correlation(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Tie-stable Spearman correlation without adding a statistics dependency."""
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        result[order] = np.arange(len(values), dtype=float)
        return result

    predicted_ranks, truth_ranks = ranks(predicted), ranks(truth)
    denominator = np.linalg.norm(predicted_ranks - predicted_ranks.mean()) * np.linalg.norm(
        truth_ranks - truth_ranks.mean()
    )
    return float(np.dot(predicted_ranks - predicted_ranks.mean(), truth_ranks - truth_ranks.mean()) / denominator) if denominator else 0.0
