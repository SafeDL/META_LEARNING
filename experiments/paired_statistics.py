"""ADS-level paired summaries for the registered finite simulation cohort."""

import math

import numpy as np

from .k_shot import finite_mean


def _ads_means(values):
    return finite_mean(finite_mean(values, axis=2), axis=1)


def _bootstrap_summary(ads_values, draws):
    point = finite_mean(ads_values, axis=0)
    bootstrap = np.empty((len(draws), *point.shape))
    for start in range(0, len(draws), 128):
        bootstrap[start : start + 128] = finite_mean(ads_values[draws[start : start + 128]], axis=1)
    flat = bootstrap.reshape(len(draws), -1)
    counts = np.isfinite(flat).sum(axis=0)
    low, high = np.full(flat.shape[1], np.nan), np.full(flat.shape[1], np.nan)
    valid = counts > 0
    if valid.any():
        low[valid], high[valid] = np.nanquantile(flat[:, valid], (0.025, 0.975), axis=0)
    return {
        "mean": point,
        "ci_low": low.reshape(point.shape),
        "ci_high": high.reshape(point.shape),
        "valid_ads": np.isfinite(ads_values).sum(axis=0),
        "valid_bootstrap_replicates": counts.reshape(point.shape),
    }


def _counts(valid):
    tasks = valid.any(axis=2)
    return {
        "valid_tasks": tasks.sum(axis=(0, 1)),
        "valid_seed_runs": valid.sum(axis=(0, 1, 2)),
        "valid_scenes_per_ads": tasks.sum(axis=1),
    }


def _leave_groups_out(ads_values, labels):
    labels = np.asarray(labels)
    return {
        label: {
            "mean": finite_mean(ads_values[labels != label], axis=0),
            "valid_ads": np.isfinite(ads_values[labels != label]).sum(axis=0),
        }
        for label in dict.fromkeys(labels.tolist())
    }


def json_ready(value):
    """Undefined statistics become JSON null, preserving observed zeros."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def paired_summary(
    values,
    *,
    ads_ids,
    ads_families,
    methods,
    baseline="target_only",
    lower_better=True,
    replicates=10000,
    seed=1301,
):
    """ADS × scene × feedback seed × method × optional curve/metric coordinates.

    Undefined metrics are NaN. Each comparison first restricts both methods to
    the same valid seed cells; then averages seeds, scenes, and finally ADSs.
    Bootstrap draws resample whole ADSs, keeping every method/curve pair intact.
    Intervals describe this registered cohort; they are pointwise, not tests.
    """
    values = np.asarray(values, dtype=float)
    ads_ids, ads_families, methods = list(ads_ids), list(ads_families), list(methods)
    if (
        values.ndim < 4
        or not all(values.shape)
        or len(ads_ids) != values.shape[0]
        or len(ads_families) != values.shape[0]
        or len(set(ads_ids)) != len(ads_ids)
        or len(methods) != values.shape[3]
        or len(set(methods)) != len(methods)
        or baseline not in methods
        or any(not label for label in (*ads_ids, *ads_families, *methods))
        or np.isinf(values).any()
        or type(replicates) is not int
        or replicates < 1
    ):
        raise ValueError(
            "Expected aligned, nonempty ADS/scene/seed/method cells and finite or undefined metrics"
        )
    direction = np.broadcast_to(np.asarray(lower_better, dtype=bool), values.shape[4:])
    baseline_index = methods.index(baseline)
    reference = np.broadcast_to(values[:, :, :, baseline_index : baseline_index + 1], values.shape)
    valid = np.isfinite(values)
    common = valid & np.isfinite(reference)
    own_ads = _ads_means(values)
    method_ads = _ads_means(np.where(common, values, np.nan))
    reference_ads = _ads_means(np.where(common, reference, np.nan))
    gains = np.where(direction, reference_ads - method_ads, method_ads - reference_ads)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(ads_ids), size=(replicates, len(ads_ids)))
    unpaired = {
        **_bootstrap_summary(own_ads, draws),
        **_counts(valid),
        "ads_values": own_ads,
    }
    paired = _bootstrap_summary(gains, draws)
    paired = {
        "gain": paired["mean"],
        "gain_ci_low": paired["ci_low"],
        "gain_ci_high": paired["ci_high"],
        "valid_ads": paired["valid_ads"],
        "valid_bootstrap_replicates": paired["valid_bootstrap_replicates"],
        "reference_mean": finite_mean(reference_ads, axis=0),
        "method_mean": finite_mean(method_ads, axis=0),
        "ads_gain": gains,
        **_counts(common),
    }
    return {
        "design": {
            "ads_ids": ads_ids,
            "ads_families": ads_families,
            "methods": methods,
            "baseline": baseline,
            "registered_ads": len(ads_ids),
            "registered_tasks": values.shape[0] * values.shape[1],
            "registered_seed_runs": int(np.prod(values.shape[:3])),
            "aggregation": "feedback seeds within scene, scenes within ADS, equal ADS mean",
            "pairing": "common defined seed cells before any averaging",
            "bootstrap_unit": "ADS only; no feedback-seed resampling",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "confidence_level": 0.95,
            "interval_type": "pointwise percentile; valid replicate counts reported",
            "positive_gain": "method improves on the baseline in the supplied metric direction",
            "scope": "descriptive uncertainty for the registered ADS configurations",
            "leave_group_out_retrains_models": False,
        },
        "unpaired": unpaired,
        "paired": paired,
        "leave_ads_out": _leave_groups_out(gains, ads_ids),
        "leave_family_out": _leave_groups_out(gains, ads_families),
    }
