"""Summarize complete, evaluator-owned formal metric grids without refitting models."""

import hashlib
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np

from .k_shot import CHECKPOINTS, LOWER_BETTER, METRICS
from .paired_statistics import json_ready, paired_summary


def _grid(rows, ads, scenes, seeds, methods, levels):
    axes = [list(ads), list(scenes), list(seeds), list(methods), list(levels)]
    if any(not axis or len(set(axis)) != len(axis) for axis in axes):
        raise ValueError("A summary requires nonempty, unique registered cohort axes")
    expected = set(product(*axes))
    lookup = {}
    for row in rows:
        key = row["sut_id"], row["scenario"], row["seed"], row["method"], row.get("K")
        if key in lookup or key not in expected:
            raise ValueError("Duplicate or unregistered metric row")
        if row["status"] not in ("complete", "failed"):
            raise ValueError("Pending runs cannot enter final statistical summaries")
        if row["status"] == "failed" and not row.get("failure_reason"):
            raise ValueError("Failed runs require their recorded reason")
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError("Incomplete registered metric grid; do not summarize a partial cohort")
    return axes, lookup


def _pool_counts(rows, axes, candidate_count):
    ads, scenes = axes[:2]
    pool = np.full((len(ads), len(scenes)), -1, dtype=int)
    for row in rows:
        count = row["pool_collisions"]
        if (
            type(count) is not int
            or not 0 <= count <= candidate_count
            or row["candidate_count"] != candidate_count
        ):
            raise ValueError(
                "Full-library collision counts must match the registered candidate count"
            )
        a, s = ads.index(row["sut_id"]), scenes.index(row["scenario"])
        if pool[a, s] not in (-1, count):
            raise ValueError("Full-library truth changed across methods, seeds or checkpoints")
        pool[a, s] = count
    return pool


def _summaries(values, pool, axes, ads, lower_better, replicates):
    kwargs = {
        "ads_ids": axes[0],
        "ads_families": [ads[name] for name in axes[0]],
        "methods": axes[3],
        "lower_better": lower_better,
        "replicates": replicates,
    }
    result = {"all": paired_summary(values, **kwargs)}
    for index, scene in enumerate(axes[1]):
        result[scene] = paired_summary(values[:, index : index + 1], **kwargs)
    expansion = (slice(None), slice(None), *(None for _ in values.shape[2:]))
    for name, condition in (
        ("zero_pool_collisions", pool == 0),
        ("nonzero_pool_collisions", pool > 0),
    ):
        result[name] = paired_summary(np.where(condition[expansion], values, np.nan), **kwargs)
    return result


def prediction_summary(
    rows,
    *,
    ads,
    scenes,
    seeds,
    methods,
    checkpoints=CHECKPOINTS,
    candidate_count=2000,
    replicates=10000,
):
    if any(type(k) is not int or k < 0 for k in checkpoints) or max(checkpoints) >= candidate_count:
        raise ValueError("Registered feedback checkpoints must fit the candidate library")
    axes, lookup = _grid(rows, ads, scenes, seeds, methods, checkpoints)
    pool = _pool_counts(rows, axes, candidate_count)
    values = np.full(tuple(map(len, axes)) + (len(METRICS),), np.nan)
    undefined, failed, evaluation_truth = [], [], {}
    reasons = {
        "recall_100": "recall_undefined",
        "auroc": "auroc_undefined",
        "auprc_ap": "auprc_undefined",
    }
    for indices in np.ndindex(*map(len, axes)):
        key = tuple(axis[index] for axis, index in zip(axes, indices))
        row = lookup[key]
        if row["status"] == "failed":
            if any(row.get(metric) is not None for metric in METRICS):
                raise ValueError("A failed prediction cannot contain completed metric values")
            failed.append(
                {k: row[k] for k in ("sut_id", "scenario", "seed", "method", "K", "failure_reason")}
            )
            continue
        evaluation_count, collisions = row["evaluation_count"], row["evaluation_collisions"]
        full_count = pool[indices[:2]]
        if (
            evaluation_count != candidate_count - max(checkpoints)
            or type(collisions) is not int
            or not max(0, full_count - max(checkpoints))
            <= collisions
            <= min(full_count, evaluation_count)
        ):
            raise ValueError("Invalid fixed-evaluation truth counts")
        truth_key = row["sut_id"], row["scenario"], row["seed"]
        truth = evaluation_count, collisions
        if evaluation_truth.get(truth_key, truth) != truth:
            raise ValueError("Evaluation truth differs across K or methods")
        evaluation_truth[truth_key] = truth
        for metric in ("recall_100", "auprc_ap"):
            if (row[metric] is None) != (collisions == 0):
                raise ValueError("Undefined ranking metrics must follow fixed-evaluation positives")
        if (row["auroc"] is None) != (collisions in (0, evaluation_count)):
            raise ValueError("AUROC requires both evaluation classes")
        for i, metric in enumerate(METRICS):
            value = row[metric]
            if value is None:
                reason = row.get(reasons.get(metric))
                if not reason:
                    raise ValueError(
                        "Undefined metric requires its reason; defined metrics cannot be missing"
                    )
                undefined.append({"key": key, "metric": metric, "reason": reason})
            elif not np.isfinite(value) or value < 0 or (metric != "nll" and value > 1 + 1e-12):
                raise ValueError("Invalid prediction metric value")
            else:
                values[(*indices, i)] = value
    result = {
            "experiment": "strict_k_shot",
            "metrics": list(METRICS),
            "scenes": axes[1],
            "seeds": axes[2],
            "checkpoints": axes[4],
            "candidate_count": candidate_count,
            "failed_rows": failed,
            "undefined_metric_rows": undefined,
            "scopes": _summaries(values, pool, axes, ads, LOWER_BETTER, replicates),
        }
    few = (5, 10, 20, 50)
    if set(few).issubset(checkpoints) and "full" in methods:
        def scopes(array, method_names, baseline):
            kwargs = {
                "ads_ids": axes[0], "ads_families": [ads[name] for name in axes[0]],
                "methods": method_names, "baseline": baseline,
                "lower_better": LOWER_BETTER, "replicates": replicates,
            }
            return {
                "all": paired_summary(array, **kwargs),
                **{scene: paired_summary(array[:, index:index + 1], **kwargs) for index, scene in enumerate(axes[1])},
            }

        indices = [axes[4].index(k) for k in few]
        averaged = np.mean(values[:, :, :, :, indices, :], axis=4)
        primary = scopes(averaged, axes[3], "target_only")
        ours_index = axes[3].index("full")
        judgment = {}
        for scope, summary in primary.items():
            gain = summary["paired"]["gain"][ours_index]
            lower = summary["paired"]["gain_ci_low"][ours_index]
            supported = {
                metric: bool(gain[METRICS.index(metric)] > 0 and lower[METRICS.index(metric)] > 0)
                for metric in ("nll", "recall_100")
            }
            judgment[scope] = {**supported, "both_supported": all(supported.values())}
        ours = values[:, :, :, ours_index]
        initial = np.repeat(ours[:, :, :, axes[4].index(0):axes[4].index(0) + 1], len(axes[4]), axis=3)
        result["few_feedback_analysis"] = {
            "checkpoints": list(few),
            "aggregation": "equal mean of the four K checkpoints within each ADS-scene; then equal ADS mean",
            "reference": "target_only", "scopes": primary, "support": judgment,
            "comparison_with_pooled": scopes(averaged, axes[3], "pooled_history") if "pooled_history" in methods else None,
        }
        result["adaptation_from_own_k0"] = {
            "methods": ["initial_k0", "adapted"],
            "gain_direction": "NLL/Brier: K=0 minus K; Recall@100: K minus K=0",
            "scopes": scopes(np.stack((initial, ours), axis=3), ["initial_k0", "adapted"], "initial_k0"),
        }
    return json_ready(result)


def budget_summary(
    rows,
    *,
    ads,
    scenes,
    seeds,
    methods,
    budget=200,
    initial_count=20,
    candidate_count=2000,
    replicates=10000,
):
    if not 0 < initial_count < budget <= candidate_count:
        raise ValueError("Registered initialization and budget must fit the candidate library")
    axes, lookup = _grid(rows, ads, scenes, seeds, methods, (None,))
    pool = _pool_counts(rows, axes, candidate_count)
    shape = tuple(map(len, axes[:4]))
    curves = np.full((*shape, budget, 2), np.nan)
    endpoints = np.full((*shape, 3), np.nan)
    failed, initial_curves = [], {}
    for index in np.ndindex(*shape):
        key = (*[axis[i] for axis, i in zip(axes[:4], index)], None)
        row = lookup[key]
        if row["status"] == "failed":
            failed.append(
                {k: row[k] for k in ("sut_id", "scenario", "seed", "method", "failure_reason")}
            )
            continue
        metrics = row["metrics"]
        curve = np.asarray(metrics["discovery_curve"], dtype=float)
        count = pool[index[:2]]
        if (
            metrics["queries"] != budget
            or metrics["initial_count"] != initial_count
            or metrics["candidate_count"] != candidate_count
            or metrics["pool_unique_collisions"] != count
            or curve.shape != (budget,)
            or not np.isfinite(curve).all()
            or (curve < 0).any()
            or (curve > count).any()
            or (curve != np.floor(curve)).any()
            or ((np.diff(np.r_[0, curve]) < 0) | (np.diff(np.r_[0, curve]) > 1)).any()
            or metrics["unique_collisions"] != curve[-1]
            or metrics["initial_collisions"] != curve[initial_count - 1]
            or metrics["post_initial_collisions"] != curve[-1] - curve[initial_count - 1]
            or not np.isfinite(metrics["discovery_auc"])
            or abs(metrics["discovery_auc"] - curve.mean()) > 1e-12
        ):
            raise ValueError("Budget metrics do not match the registered unique-discovery curve")
        initial_key = row["sut_id"], row["scenario"], row["seed"]
        prefix = tuple(curve[:initial_count])
        if initial_curves.get(initial_key, prefix) != prefix:
            raise ValueError("Methods do not share the same initial discovery curve")
        initial_curves[initial_key] = prefix
        recall = np.asarray([np.nan if r is None else r for r in metrics["recall_curve"]])
        expected = curve / count if count else np.full(budget, np.nan)
        if recall.shape != expected.shape or not np.allclose(
            recall, expected, rtol=0, atol=1e-12, equal_nan=True
        ):
            raise ValueError("Coverage must retain undefined zero-collision tasks")
        final_recall = metrics["collision_recall"]
        if (count == 0 and final_recall is not None) or (
            count
            and (
                final_recall is None
                or not np.isfinite(final_recall)
                or abs(final_recall - expected[-1]) > 1e-12
            )
        ):
            raise ValueError("Final coverage differs from its full-library denominator")
        curves[index] = np.stack((curve, expected), axis=-1)
        endpoints[index] = curve.mean(), curve[-1], curve[-1] - curve[initial_count - 1]
    # Supplementary weighted fraction: use exactly the same completed runs for all methods.
    common = np.isfinite(curves[..., 0]).all(axis=-1).all(axis=-1)
    denominator = int(np.broadcast_to(pool[:, :, None], common.shape)[common].sum())
    numerator = np.nansum(np.where(common[..., None, None], curves[..., 0], np.nan), axis=(0, 1, 2))
    return json_ready(
        {
            "experiment": "budgeted_discovery",
            "budget": budget,
            "initial_count": initial_count,
            "candidate_count": candidate_count,
            "scenes": axes[1],
            "seeds": axes[2],
            "curve_metrics": ["cumulative_unique_collisions", "collision_coverage"],
            "endpoint_metrics": [
                "discovery_auc",
                "final_unique_collisions",
                "post_initial_collisions",
            ],
            "failed_runs": failed,
            "curves": _summaries(curves, pool, axes, ads, False, replicates),
            "endpoints": _summaries(endpoints, pool, axes, ads, False, replicates),
            "pooled_run_coverage": {
                "common_completed_seed_tasks": int(common.sum()),
                "full_pool_collision_denominator": denominator,
                "discovery_numerator": numerator if common.any() else None,
                "coverage_curve": numerator / denominator
                if denominator
                else np.full(numerator.shape, np.nan),
                "definition": "Sum discoveries over common completed logical runs divided by the corresponding sum of full-pool collisions; seeds are repeated runs, not unique global defects",
            },
        }
    )


def register_statistics(root=None):
    from controllers.highway.catalog import FORMAL_TARGET_SUTS

    from .history_transfer_experiment import (
        CHECKPOINTS,
        OUTPUT,
        PREDICTION_METHODS,
        SEEDS,
        TARGET_COUNT,
        freeze,
        scenarios_for,
    )
    from .protocol import ROOT

    root = OUTPUT if root is None else root
    scenes = scenarios_for(root)
    ads = {s.sut_id: s.family for s in FORMAL_TARGET_SUTS}
    task_seeds = len(ads) * len(scenes) * len(SEEDS)
    paths = (
        "experiments/history_transfer_summary.py",
        "experiments/paired_statistics.py",
        "experiments/k_shot.py",
        "experiments/failure_metrics.py",
    )
    return freeze(
        Path(OUTPUT if root is None else root) / "reports/statistics_protocol.json",
        {
            "scope": "Preregistered descriptive analysis of the finite formal target cohort",
            "ads": ads,
            "scenarios": list(scenes),
            "feedback_seeds": list(SEEDS),
            "candidate_count": TARGET_COUNT,
            "prediction": {
                "methods": list(PREDICTION_METHODS),
                "checkpoints": list(CHECKPOINTS),
                "evaluation_count": TARGET_COUNT - max(CHECKPOINTS),
                "metrics": list(METRICS),
                "lower_better": LOWER_BETTER.tolist(),
                "trajectories": task_seeds * len(PREDICTION_METHODS),
                "metric_rows": task_seeds * len(PREDICTION_METHODS) * len(CHECKPOINTS),
                "missing": "Failed predictions have no metric values; undefined ranking metrics retain their truth-based reasons",
            },
            "statistics": {
                "aggregation": "seeds within scene; scenes within ADS; equal ADS mean",
                "pairing": "same defined seed cells before averaging either method",
                "bootstrap_unit": "ADS; all methods, scenes, seeds and curve coordinates move together",
                "replicates": 10000,
                "seed": 1301,
                "confidence_level": 0.95,
                "interval": "pointwise percentile; report valid ADS, task, seed-run and bootstrap counts",
                "undefined_resamples": "exclude only resamples with no defined statistic and report their number",
                "sensitivities": "leave one ADS or algorithm family out of the summary; no retraining",
                "subgroups": "complete pool has zero or nonzero collisions; retain all tasks in the primary summary",
                "inference_scope": "variation of registered configurations; not a random-population guarantee",
                "multiplicity": "no significance testing here; multiple-K significance claims require a separately registered analysis",
            },
            "source_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
            },
            "packages": {p: version(p) for p in ("numpy", "scikit-learn")},
            "formal_target_labels_read": False,
            "new_physical_executions": 0,
        },
    )


if __name__ == "__main__":
    register_statistics()
