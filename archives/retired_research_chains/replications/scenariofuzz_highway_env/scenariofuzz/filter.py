"""SEM threshold selection and explicit cold-start policies."""

from __future__ import annotations

import numpy as np
import torch

from .corpus import LocalScenarioSeed, ScenarioSpec
from .graph_builder import build_graph, stack_graphs
from .sem_model import ScenarioEvaluationModel


def predict_scores(model: ScenarioEvaluationModel, seed: LocalScenarioSeed, candidates: list[ScenarioSpec], device: str = "cpu") -> np.ndarray:
    model.eval()
    batch = stack_graphs([build_graph(seed, spec) for spec in candidates], device)
    with torch.no_grad():
        return torch.sigmoid(model(batch)).cpu().numpy()


def select_candidates(candidates: list[ScenarioSpec], scores: np.ndarray | None, ne: int, threshold: float, rng: np.random.Generator, empty_policy: str = "resample") -> tuple[list[int], str]:
    if scores is None:
        size = min(ne, len(candidates))
        return rng.choice(len(candidates), size=size, replace=False).tolist(), "cold_start_random"
    scores = np.asarray(scores, dtype=float)
    eligible = np.flatnonzero(scores > threshold)
    if len(eligible):
        order = eligible[np.argsort(scores[eligible])[::-1]]
        return order[:ne].tolist(), "threshold_descending"
    if empty_policy == "fill_topk":
        return np.argsort(scores)[::-1][: min(ne, len(scores))].tolist(), "fill_topk_extension"
    return [], "empty_filter_resample"


def load_checkpoint(path, device: str = "cpu") -> tuple[ScenarioEvaluationModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ScenarioEvaluationModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint

