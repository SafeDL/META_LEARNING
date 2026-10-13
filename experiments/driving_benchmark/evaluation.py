"""Frozen collision-discovery scoring shared by baseline evaluation."""
import numpy as np

from .config import BUDGET, CHECKPOINTS


def metrics(collisions, selected):
    positives = int(np.count_nonzero(collisions))
    found = np.cumsum(np.asarray(collisions)[selected])
    upper = np.minimum(np.arange(1, BUDGET + 1), positives)
    return {
        "total_collisions": positives,
        "area_200": float(found.sum() / (BUDGET * positives)) if positives else None,
        "recall_200": float(found[-1] / positives) if positives else None,
        "F200": int(found[-1]), "missed": positives - int(found[-1]),
        "checkpoints": {str(t): int(found[t - 1]) for t in CHECKPOINTS},
        "terminal_ceiling_attainment": float(found[-1] / upper[-1]) if positives else None,
        "area_ceiling": float(upper.sum() / (BUDGET * positives)) if positives else None,
        "curve": found.astype(int).tolist(),
    }
