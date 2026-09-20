"""Offline-only discovery metrics for a fixed candidate pool."""

from __future__ import annotations

import numpy as np


def discovery_metrics(order: list[int], failed: np.ndarray,
                      critical: np.ndarray) -> dict[str, object]:
    """Score an already chosen order without exposing labels to the selector."""
    selected = np.asarray(order, dtype=int)
    failure_values, critical_values = np.asarray(failed, dtype=bool)[selected], np.asarray(
        critical, dtype=bool)[selected]
    total_failures = int(np.asarray(failed, dtype=bool).sum())
    return {
        "selected_count": int(len(selected)),
        "failure_count": int(failure_values.sum()),
        "failure_ratio": float(failure_values.mean()) if len(selected) else float("nan"),
        "recall": float(failure_values.sum() / total_failures) if total_failures else float("nan"),
        "critical_count": int(critical_values.sum()),
        "failure_curve": np.cumsum(failure_values, dtype=int).tolist(),
        "critical_curve": np.cumsum(critical_values, dtype=int).tolist()
    }
