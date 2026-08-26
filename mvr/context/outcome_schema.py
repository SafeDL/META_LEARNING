"""Explicit mixed binary/continuous target contract for posterior learning."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F


OUTCOME_FIELDS = ("failure", "invalid", "normalized_min_ttc", "normalized_min_distance", "normalized_max_closing_speed")


def encode_outcome(outcome: Mapping[str, Any], *, ttc_scale_s: float = 15.0, distance_scale_m: float = 100.0, closing_speed_scale_mps: float = 30.0) -> torch.Tensor:
    """Convert raw rollout outcomes into the fixed target order used by the decoder."""
    invalid = not bool(outcome.get("is_valid_episode", not any(bool(outcome.get(key, False)) for key in ("non_target_collision", "adversary_out_of_road", "sut_out_of_road", "wrong_route", "adversary_traffic_violation"))))
    failure = bool(outcome.get("is_failure", not invalid and (outcome.get("target_collision", False) or outcome.get("valid_critical_near_miss", outcome.get("valid_critical_strict", False)))))
    values = (float(failure), float(invalid), float(outcome.get("min_ttc", ttc_scale_s)) / ttc_scale_s,
              float(outcome.get("min_distance", distance_scale_m)) / distance_scale_m,
              float(outcome.get("max_closing_speed", outcome.get("closing_speed_mps", 0.0))) / closing_speed_scale_mps)
    return torch.tensor(values, dtype=torch.float32).clamp(0.0, 1.0)


def outcome_elbo(prediction: torch.Tensor, target: torch.Tensor, *, kl: torch.Tensor, severity_weight: float = 1.0, kl_weight: float = 1e-3) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != len(OUTCOME_FIELDS):
        raise ValueError("outcome decoder requires aligned five-dimensional predictions and targets")
    binary = F.binary_cross_entropy_with_logits(prediction[..., :2], target[..., :2].float())
    severity = F.huber_loss(prediction[..., 2:], target[..., 2:].float())
    return binary + float(severity_weight) * severity + float(kl_weight) * kl.mean()
