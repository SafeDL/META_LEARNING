"""Common feedback partition, prediction metrics and K-shot result validation."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CHECKPOINTS = (0, 5, 10, 20, 50, 100, 200)
FEEDBACK_COUNT = max(CHECKPOINTS)
METRICS = ("nll", "brier", "recall_100", "auroc", "auprc_ap", "ece")
LOWER_BETTER = np.array([True, True, False, False, False, True])


def split_candidates(candidates, seed, *, budget=FEEDBACK_COUNT):
    ids = [s.scenario_id for s in candidates]
    if len(set(ids)) != len(ids) or not 0 < budget < len(ids):
        raise ValueError("A split needs unique IDs and nonempty disjoint query/evaluation sets")
    canonical = sorted(range(len(ids)), key=ids.__getitem__)
    permutation = np.random.default_rng(seed).permutation(len(ids))
    order = [canonical[int(i)] for i in permutation]
    return {
        "candidate_ids": ids,
        "query_order": order[:budget],
        "evaluation_indices": order[budget:],
    }


def validate_result(result, split, *, checkpoints=CHECKPOINTS):
    order = split["query_order"]
    if result["selected_indices"] != order or len(result["queries"]) != len(order):
        raise ValueError("Method did not replay the complete common feedback sequence")
    snapshots = result["prediction_snapshots"]
    if [s["paid_queries"] for s in snapshots] != list(checkpoints):
        raise ValueError("Missing, duplicate or misplaced K-shot snapshots")
    evaluation = split["evaluation_indices"]
    if set(evaluation) & set(order) or len(evaluation) + len(order) != len(split["candidate_ids"]):
        raise ValueError("Query and fixed evaluation sets must partition the candidate bank")
    for snapshot in snapshots:
        count = snapshot["paid_queries"]
        if snapshot["observed_indices"] != order[:count]:
            raise ValueError("Snapshot used a different target feedback sequence")
        p = np.asarray(snapshot["probabilities"])
        if (
            p.shape != (len(split["candidate_ids"]),)
            or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()
        ):
            raise ValueError("Snapshot contains invalid collision probabilities")
        if not all(np.isfinite(v) for v in snapshot["target_parameters"].values()):
            raise ValueError("Snapshot contains nonfinite fitted parameters")


def prediction_metrics(
    probabilities, labels, candidate_ids, evaluation_indices, *, top=(50, 100, 200)
):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    indices = np.asarray(evaluation_indices, dtype=int)
    if p.shape != y.shape or p.shape != (len(candidate_ids),):
        raise ValueError("Predictions, labels and candidate IDs must align")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Collision probabilities must be finite and within [0, 1]")
    if not np.isin(y, (0, 1)).all() or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Binary labels and unique candidate IDs are required")
    if (
        not len(indices)
        or len(set(indices)) != len(indices)
        or ((indices < 0) | (indices >= len(p))).any()
    ):
        raise ValueError("The fixed evaluation set must contain unique valid indices")
    p, y = p[indices], y[indices]
    ids = np.asarray(candidate_ids)[indices]
    positive_count = int(y.sum())
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    metrics = {
        "evaluation_count": len(y),
        "evaluation_collisions": positive_count,
        "nll": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log(1 - bounded))),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": float(roc_auc_score(y, p)) if 0 < positive_count < len(y) else None,
        "auroc_undefined": None if 0 < positive_count < len(y) else "single_class",
        "auprc_ap": float(average_precision_score(y, p)) if positive_count else None,
        "auprc_undefined": None if positive_count else "no_positive_labels",
    }
    ranking = np.lexsort((ids, -p))
    for limit in top:
        if limit < 1:
            raise ValueError("Recall cutoffs must be positive")
        actual = min(limit, len(y))
        hits = int(y[ranking[:actual]].sum())
        metrics.update(
            {
                f"top_{limit}_count": actual,
                f"top_{limit}_hits": hits,
                f"recall_{limit}": hits / positive_count if positive_count else None,
            }
        )
    metrics["recall_undefined"] = None if positive_count else "no_positive_labels"
    bins = np.minimum((10 * p).astype(int), 9)
    calibration = []
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        calibration.append(
            {
                "bin": index,
                "count": count,
                "mean_probability": float(p[mask].mean()) if count else None,
                "collision_fraction": float(y[mask].mean()) if count else None,
            }
        )
    metrics["ece"] = sum(
        b["count"] / len(y) * abs(b["mean_probability"] - b["collision_fraction"])
        for b in calibration
        if b["count"]
    )
    return metrics, calibration


def finite_mean(values, axis=None):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
