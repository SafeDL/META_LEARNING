"""Reference-distribution and provenance helpers for finite candidate pools."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def scenario_ids(anchors: np.ndarray, modes: Iterable[str]) -> list[str]:
    """Return stable IDs from canonical physical parameters and mode."""
    values = np.asarray(anchors, dtype=float)
    labels = [str(mode) for mode in modes]
    if values.ndim != 2 or values.shape[1] != 2 or len(labels) != len(values):
        raise ValueError("anchors must be [L,2] and modes must contain L values")
    identifiers: list[str] = []
    for (gap, relative_speed), mode in zip(values, labels):
        canonical = f"{mode}|gap={gap:.9f}|relative_speed={relative_speed:.9f}"
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        identifiers.append(f"fst-{digest}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scenario IDs are not unique")
    return identifiers


def validate_distribution(probability: np.ndarray, length: int) -> np.ndarray:
    """Validate a finite probability mass function without silently renormalizing."""
    p = np.asarray(probability, dtype=float)
    if p.shape != (length,):
        raise ValueError(f"reference distribution must have shape ({length},)")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("reference distribution must be finite and non-negative")
    if not np.isclose(float(p.sum()), 1.0, atol=1e-10):
        raise ValueError(f"reference distribution sums to {p.sum():.16g}, not 1")
    return p


def uniform_distribution(length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("candidate pool must be non-empty")
    return np.full(length, 1.0 / length, dtype=float)


def write_distribution(path: Path, identifiers: list[str], probability: np.ndarray) -> None:
    p = validate_distribution(probability, len(identifiers))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "probability"])
        writer.writerows((identifier, f"{mass:.17g}") for identifier, mass in zip(identifiers, p))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

