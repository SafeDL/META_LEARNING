"""Prediction diagnostics on identical disclosed support; no model fitting."""
import numpy as np
from sklearn.metrics import average_precision_score


def prediction_metrics(mean, variance, z, labels, mask):
    error = mean[mask] - z[mask]
    sd = np.sqrt(variance[mask] + 0.001)
    covered = np.abs(error) <= 1.96 * sd
    target = labels[mask]
    top = np.argsort(-mean[mask], kind="stable")[:50]
    return {
        "remaining_count": int(mask.sum()),
        "remaining_collisions": int(target.sum()),
        "bias_z": float(error.mean()),
        "rmse_z": float(np.sqrt(np.mean(error ** 2))),
        "rmse_z_collision": float(np.sqrt(np.mean(error[target == 1] ** 2)))
        if target.sum() else None,
        "bias_z_collision": float(error[target == 1].mean()) if target.sum() else None,
        "collision_ap": float(average_precision_score(target, mean[mask]))
        if 0 < target.sum() < len(target) else None,
        "top50_collisions": int(target[top].sum()),
        "mean_latent_var": float(variance[mask].mean()),
        "coverage95_all": float(covered.mean()),
        "coverage95_collision": float(covered[target == 1].mean())
        if target.sum() else None,
    }
