"""Lineage-weighted summaries shared by the retained experiments."""

import numpy as np


def summary(rows, metric):
    lineages = sorted({r["lineage"] for r in rows})
    grouped = {
        lineage: [r[metric] for r in rows if r["lineage"] == lineage and r[metric] is not None]
        for lineage in lineages
    }
    values = {name: float(np.mean(items)) for name, items in grouped.items() if items}
    return {
        "mean": float(np.mean(list(values.values()))) if values else None,
        "by_lineage": values,
        "valid_lineages": len(values),
        "valid_runs": sum(len(items) for items in grouped.values()),
    }


def method_summary(rows):
    return {
        method: {
            metric: summary([r for r in rows if r["method"] == method], metric)
            for metric in (
                "unique_collisions",
                "collision_recall",
                "discovery_auc",
                "selection_s",
                "adaptation_s",
            )
        }
        for method in sorted({r["method"] for r in rows})
    }
