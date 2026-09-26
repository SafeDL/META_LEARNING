"""Correlated Gaussian priors for the FBRT logistic pattern model (v3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from method_chains.failure_memory_regression.pattern_memory import RBFDictionary
from method_chains.failure_memory_regression.replay_utils import is_usable_outcome


@dataclass
class LogisticFit:
    mean: np.ndarray
    covariance: np.ndarray
    iterations: int
    converged: bool


def _as_covariance(value: np.ndarray | None, dim: int,
                   default_variance: float = 4.0) -> np.ndarray:
    if value is None:
        covariance = np.eye(dim) * default_variance
    else:
        value = np.asarray(value, dtype=float)
        covariance = np.diag(value) if value.ndim == 1 else value.copy()
    if covariance.shape != (dim, dim) or not np.all(np.isfinite(covariance)):
        raise ValueError("prior covariance dimensions or values are invalid")
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return covariance


def _inverse_spd(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    matrix = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) * 0.5
    eye = np.eye(matrix.shape[0])
    for scale in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            chol = np.linalg.cholesky(matrix + eye * jitter * scale)
            inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
            return (inverse + inverse.T) * 0.5
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(matrix + eye * jitter * 100000)


def _objective(theta: np.ndarray, features: np.ndarray, labels: np.ndarray,
               prior_mean: np.ndarray, prior_precision: np.ndarray) -> float:
    delta = theta - prior_mean
    value = 0.5 * float(delta @ prior_precision @ delta)
    if len(labels):
        logits = features @ theta
        value += float(np.sum(np.logaddexp(0.0, logits) - labels * logits))
    return value


def fit_logistic(features: np.ndarray, labels: np.ndarray,
                 prior_mean: np.ndarray | None = None,
                 prior_variance: np.ndarray | None = None,
                 max_iter: int = 25, jitter: float = 1e-8,
                 tolerance: float = 1e-7,
                 fit_counter: dict[str, int] | None = None) -> LogisticFit:
    if fit_counter is not None:
        fit_counter["model_fit_count"] = int(fit_counter.get("model_fit_count", 0)) + 1
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or (len(labels) and features.shape[0] != len(labels)):
        raise ValueError("features must be N×D and labels length N")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(labels)):
        raise ValueError("features and labels must be finite")
    dim = features.shape[1]
    mean0 = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    if mean0.shape != (dim,) or not np.all(np.isfinite(mean0)):
        raise ValueError("prior mean dimensions or values are invalid")
    covariance0 = _as_covariance(prior_variance, dim)
    precision0 = _inverse_spd(covariance0, jitter)
    theta = mean0.copy()
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        probabilities = expit(features @ theta) if len(labels) else np.empty(0)
        gradient = precision0 @ (theta - mean0)
        hessian = precision0.copy()
        if len(labels):
            gradient += features.T @ (probabilities - labels)
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            hessian += features.T @ (weights[:, None] * features)
        step = _inverse_spd(hessian, jitter) @ gradient
        old_loss = _objective(theta, features, labels, mean0, precision0)
        scale = 1.0
        candidate = theta - step
        while scale > 1.0 / 128 and _objective(
                candidate, features, labels, mean0, precision0) > old_loss + 1e-10:
            scale *= 0.5
            candidate = theta - scale * step
        if np.linalg.norm(candidate - theta) <= tolerance * (1.0 + np.linalg.norm(theta)):
            theta = candidate
            converged = True
            break
        theta = candidate
    probabilities = expit(features @ theta) if len(labels) else np.empty(0)
    hessian = precision0.copy()
    if len(labels):
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        hessian += features.T @ (weights[:, None] * features)
    return LogisticFit(theta, _inverse_spd(hessian, jitter), iterations, converged)


def source_fits(records: list[dict], dictionary: RBFDictionary,
                fit_counter: dict[str, int] | None = None) -> dict[str, LogisticFit]:
    output = {}
    for build_id in sorted({row["build_id"] for row in records}):
        local = [row for row in records if row["build_id"] == build_id and
                 row.get("template_id") == dictionary.template_id and
                 row.get("visibility") != "evaluator_only" and is_usable_outcome(row)]
        if not local:
            continue
        x = np.vstack([dictionary.features(row["scenario"]) for row in local])
        y = np.asarray([float(row["ego_collision"]) for row in local])
        output[build_id] = fit_logistic(x, y, max_iter=25,
                                        fit_counter=fit_counter)
    return output


def _family_weights(build_ids: list[str], families: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[str]] = {}
    for build_id in build_ids:
        grouped.setdefault(families.get(build_id, build_id), []).append(build_id)
    return {build_id: 1.0 / len(grouped) / len(grouped[families.get(build_id, build_id)])
            for build_id in build_ids}


def _floor(dim: int) -> np.ndarray:
    return np.asarray([4.0, 1.0, 1.0, *([2.25] * max(0, dim - 3))])[:dim]


def build_source_prior(fits: dict[str, LogisticFit], families: dict[str, str],
                       parent_build_id: str | None = None,
                       regression: bool = False,
                       feature_ids: Sequence[str] | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not fits:
        dim = len(feature_ids) if feature_ids is not None else 0
        return np.zeros(dim), np.eye(dim) * 4.0, []
    dim = len(next(iter(fits.values())).mean)
    used = sorted(fits)
    floor = np.diag(_floor(dim))
    if regression and parent_build_id in fits:
        parent = fits[parent_build_id]
        return parent.mean.copy(), parent.covariance.copy() + floor, [parent_build_id]
    weights = _family_weights(used, families)
    mean = sum(weights[key] * fits[key].mean for key in used)
    covariance = np.zeros((dim, dim), dtype=float)
    for key in used:
        delta = fits[key].mean - mean
        covariance += weights[key] * (
            fits[key].covariance + np.outer(delta, delta))
    covariance += floor
    return mean, (covariance + covariance.T) * 0.5, used


def _align_prior(old_ids: Sequence[str], old_mean: np.ndarray,
                 old_covariance: np.ndarray, new_ids: Sequence[str],
                 default_variance: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    old_ids, new_ids = tuple(old_ids), tuple(new_ids)
    old_mean = np.asarray(old_mean, dtype=float)
    old_covariance = _as_covariance(old_covariance, len(old_ids))
    if (len(set(old_ids)) != len(old_ids) or len(set(new_ids)) != len(new_ids)
            or not set(old_ids).issubset(new_ids)
            or old_mean.shape != (len(old_ids),)):
        raise ValueError("source prior features cannot be aligned by identity")
    lookup = {feature_id: index for index, feature_id in enumerate(old_ids)}
    aligned_mean = np.zeros(len(new_ids), dtype=float)
    aligned_covariance = np.eye(len(new_ids), dtype=float) * default_variance
    indices = [(new_index, lookup[feature_id])
               for new_index, feature_id in enumerate(new_ids) if feature_id in lookup]
    new_indices = [new_index for new_index, _ in indices]
    old_indices = [old_index for _, old_index in indices]
    aligned_mean[new_indices] = old_mean[old_indices]
    aligned_covariance[np.ix_(new_indices, new_indices)] = old_covariance[
        np.ix_(old_indices, old_indices)]
    return aligned_mean, aligned_covariance


def target_posterior(dictionary: RBFDictionary, observations: list[dict],
                     source_prior_ids: Sequence[str] | np.ndarray,
                     source_prior_mean: np.ndarray,
                     source_prior_variance: np.ndarray | None = None,
                     max_iter: int = 25,
                     source_feature_specs: dict[str, dict] | None = None,
                     source_schema_identity: str | None = None,
                     fit_counter: dict[str, int] | None = None) -> LogisticFit:
    if source_prior_variance is None:
        mean = np.asarray(source_prior_ids, dtype=float)
        covariance = np.asarray(source_prior_mean, dtype=float)
        ids = tuple(dictionary.ordered_feature_ids())
        if mean.shape != (len(ids),) or covariance.shape not in {
                (len(ids),), (len(ids), len(ids))}:
            raise ValueError("unlabelled prior requires the exact current feature schema")
        source_prior_ids, source_prior_mean, source_prior_variance = ids, mean, covariance
    current_ids = tuple(dictionary.ordered_feature_ids())
    if source_schema_identity is not None and source_schema_identity != dictionary.schema_identity:
        raise ValueError("source prior schema context does not match the target dictionary")
    current_specs = dictionary.ordered_feature_specs()
    if source_feature_specs is not None:
        for feature_id in source_prior_ids:
            if source_feature_specs.get(feature_id) != current_specs.get(feature_id):
                raise ValueError(f"feature semantics changed for source-prior ID {feature_id}")
    prior_mean, prior_covariance = _align_prior(
        source_prior_ids, source_prior_mean, source_prior_variance, current_ids)
    local = [row for row in observations if row.get("template_id") == dictionary.template_id
             and is_usable_outcome(row)]
    if local:
        features = np.vstack([dictionary.features(row["scenario"]) for row in local])
        labels = np.asarray([float(row["ego_collision"]) for row in local])
    else:
        features = np.empty((0, len(prior_mean)))
        labels = np.empty(0)
    return fit_logistic(features, labels, prior_mean, prior_covariance,
                        max_iter=max_iter, fit_counter=fit_counter)


def posterior_failure_probabilities(dictionary: RBFDictionary, candidates: list[dict],
                                    fit: LogisticFit, seed: int,
                                    samples: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    covariance = _as_covariance(fit.covariance, len(fit.mean))
    try:
        chol = np.linalg.cholesky(covariance + np.eye(len(covariance)) * 1e-8)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(covariance)
        chol = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    draws = fit.mean + rng.standard_normal((samples, len(fit.mean))) @ chol.T
    features = np.vstack([dictionary.features(item.get("scenario", item))
                          for item in candidates])
    logits = np.clip(draws @ features.T, -40, 40)
    return np.mean(expit(logits), axis=0)
