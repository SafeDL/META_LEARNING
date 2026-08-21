"""Outer reward keeps novelty and invalidity at the episode timescale."""
from __future__ import annotations

from .signature import FailureSignature


def outer_reward(signature: FailureSignature, *, novel: bool, failure_weight: float = 1.0, novelty_weight: float = 1.0, severity_weight: float = 0.25, invalid_penalty: float = 1.0) -> float:
    severity = sum(signature.severity_vector) / 12.0
    return (
        failure_weight * float(signature.is_failure)
        + novelty_weight * float(signature.is_failure and novel)
        + severity_weight * severity * float(signature.is_failure)
        - invalid_penalty * float(not signature.is_valid_episode)
    )
