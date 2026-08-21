"""Outer reward keeps novelty and invalidity at the episode timescale."""
from __future__ import annotations

from .signature import FailureSignature


def outer_reward(signature: FailureSignature, *, novel: bool, invalid: bool, failure_weight: float = 1.0, novelty_weight: float = 1.0, severity_weight: float = 0.25, invalid_penalty: float = 1.0) -> float:
    severity = sum(signature.severity_vector) / 12.0
    return (
        failure_weight * float(signature.valid)
        + novelty_weight * float(signature.valid and novel)
        + severity_weight * severity * float(signature.valid)
        - invalid_penalty * float(invalid)
    )
