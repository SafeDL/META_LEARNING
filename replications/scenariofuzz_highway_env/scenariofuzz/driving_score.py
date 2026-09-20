"""Explicit highway adaptation of the unavailable DriveFuzz driving score."""

from __future__ import annotations


def margin_score(vulnerability: float) -> float:
    """Lower is worse. This is 1-vulnerability, not the DriveFuzz formula."""
    return 1.0 - float(vulnerability)

