"""Small hierarchical Bayesian logistic models over the pattern RBF dictionary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from method_chains.failure_memory_regression.pattern_memory import RBFDictionary


@dataclass
class LogisticFit:
    mean: np.ndarray
    covariance: np.ndarray
    iterations: int
    converged: bool


def _inverse_spd(matrix: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    eye = np.eye(matrix.shape[0])
    for scale in (1.0, 10.0, 100.0, 1000.0):
        try:
            chol = np.linalg.cholesky(matrix + eye * jitter * scale)
            return np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(matrix + eye * jitter * 10000)


def fit_logistic(features: np.ndarray, labels: np.ndarray,
                 prior_mean: np.ndarray | None = None,
                 prior_variance: np.ndarray | None = None,
                 max_iter: int = 25, jitter: float = 1e-6,
                 tolerance: float = 1e-7) -> LogisticFit:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or (len(labels) and features.shape[0] != len(labels)):
        raise ValueError("features must be N×D and labels length N")
    dim = features.shape[1]
    mean0 = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    variance0 = np.full(dim, 4.0) if prior_variance is None else np.asarray(prior_variance, dtype=float)
    if mean0.shape != (dim,) or variance0.shape != (dim,):
        raise ValueError("prior dimensions do not match the feature dictionary")
    variance0 = np.maximum(variance0, jitter)
    precision0 = 1.0 / variance0
    theta = mean0.copy()
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        logits = features @ theta if len(labels) else np.empty(0)
        probabilities = expit(logits)
        gradient = precision0 * (theta - mean0)
        hessian = np.diag(precision0)
        if len(labels):
            gradient += features.T @ (probabilities - labels)
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            hessian += features.T @ (weights[:, None] * features)
        step = _inverse_spd(hessian, jitter) @ gradient
        # Backtracking keeps Newton updates stable for nearly separated data.
        old_loss = _objective(theta, features, labels, mean0, variance0)
        scale = 1.0
        candidate = theta - step
        while scale > 1 / 128 and _objective(candidate, features, labels, mean0,
                                              variance0) > old_loss + 1e-10:
            scale *= 0.5
            candidate = theta - scale * step
        if np.linalg.norm(candidate - theta) <= tolerance * (1.0 + np.linalg.norm(theta)):
            theta = candidate
            converged = True
            break
        theta = candidate
    logits = features @ theta if len(labels) else np.empty(0)
    probabilities = expit(logits)
    hessian = np.diag(precision0)
    if len(labels):
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        hessian += features.T @ (weights[:, None] * features)
    covariance = _inverse_spd(hessian, jitter)
    return LogisticFit(theta, covariance, iterations, converged)


def _objective(theta: np.ndarray, features: np.ndarray, labels: np.ndarray,
               prior_mean: np.ndarray, prior_variance: np.ndarray) -> float:
    delta = theta - prior_mean
    value = 0.5 * float(np.sum(delta * delta / prior_variance))
    if len(labels):
        logits = features @ theta
        value += float(np.sum(np.logaddexp(0.0, logits) - labels * logits))
    return value


def source_fits(records: list[dict], dictionary: RBFDictionary) -> dict[str, LogisticFit]:
    output = {}
    build_ids = sorted({row["build_id"] for row in records})
    for build_id in build_ids:
        local = [row for row in records if row["build_id"] == build_id and
                 row.get("template_id") == dictionary.template_id and
                 not row.get("inconclusive") and row.get("ego_collision") in (True, False)]
        if not local:
            continue
        x = np.vstack([dictionary.features(row["scenario"]) for row in local])
        y = np.asarray([float(row["ego_collision"]) for row in local])
        output[build_id] = fit_logistic(x, y, max_iter=25)
    return output


def _family_weights(build_ids: list[str], families: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[str]] = {}
    for build_id in build_ids:
        grouped.setdefault(families.get(build_id, build_id), []).append(build_id)
    return {build_id: 1.0 / len(grouped) / len(grouped[families.get(build_id, build_id)])
            for build_id in build_ids}


def build_source_prior(fits: dict[str, LogisticFit], families: dict[str, str],
                       parent_build_id: str | None = None,
                       regression: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not fits:
        return np.zeros(3), np.full(3, 4.0), []
    dim = len(next(iter(fits.values())).mean)
    used = sorted(fits)
    if regression and parent_build_id in fits:
        parent = fits[parent_build_id]
        floor = np.asarray([4.0, 1.0, 1.0, *([2.25] * (dim - 3))])[:dim]
        variance = np.maximum(np.diag(parent.covariance), 0.0) + floor
        return parent.mean.copy(), variance, [parent_build_id]
    weights = _family_weights(used, families)
    mean = sum(weights[key] * fits[key].mean for key in used)
    between = sum(weights[key] * (fits[key].mean - mean) ** 2 for key in used)
    within = sum(weights[key] * np.diag(fits[key].covariance) for key in used)
    floor = np.asarray([4.0, 1.0, 1.0, *([2.25] * (dim - 3))])[:dim]
    variance = np.maximum(within + between + floor, 1e-6)
    return mean, variance, used


def extend_prior(prior_mean: np.ndarray, prior_variance: np.ndarray,
                 new_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Add new failure centers with N(0, 4) coefficients and zero correlation."""
    current = len(prior_mean)
    if new_dim < current:
        raise ValueError("feature dictionary cannot shrink within a session")
    if new_dim == current:
        return prior_mean, prior_variance
    return (np.pad(prior_mean, (0, new_dim - current)),
            np.pad(prior_variance, (0, new_dim - current), constant_values=4.0))


def target_posterior(dictionary: RBFDictionary, observations: list[dict],
                     prior_mean: np.ndarray, prior_variance: np.ndarray,
                     max_iter: int = 25) -> LogisticFit:
    prior_mean, prior_variance = extend_prior(
        np.asarray(prior_mean), np.asarray(prior_variance),
        1 + dictionary.feature_dim + len(dictionary.centers) + len(dictionary.coverage_centers))
    local = [row for row in observations if row.get("template_id") == dictionary.template_id
             and row.get("ego_collision") in (True, False) and not row.get("inconclusive")]
    if local:
        x = np.vstack([dictionary.features(row["scenario"]) for row in local])
        y = np.asarray([float(row["ego_collision"]) for row in local])
    else:
        x = np.empty((0, len(prior_mean)))
        y = np.empty(0)
    return fit_logistic(x, y, prior_mean, prior_variance, max_iter=max_iter)


def posterior_failure_probabilities(dictionary: RBFDictionary, candidates: list[dict],
                                    fit: LogisticFit, seed: int,
                                    samples: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    covariance = np.asarray(fit.covariance, dtype=float)
    # Numerical jitter is deterministic; an eigen fallback handles tiny roundoff.
    try:
        chol = np.linalg.cholesky(covariance + np.eye(len(covariance)) * 1e-8)
        draws = fit.mean + rng.standard_normal((samples, len(fit.mean))) @ chol.T
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(covariance)
        root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
        draws = fit.mean + rng.standard_normal((samples, len(fit.mean))) @ root.T
    features = np.vstack([dictionary.features(item.get("scenario", item)) for item in candidates])
    logits = np.clip(draws @ features.T, -40, 40)
    return np.mean(expit(logits), axis=0)
