"""Support-protected importance weights and finite-sample diagnostics."""

from __future__ import annotations
import numpy as np


def defensive_mixture(adaptive: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    adaptive, reference = np.asarray(adaptive, dtype=float), np.asarray(reference, dtype=float)
    if adaptive.shape != reference.shape or not 0 <= epsilon <= 1:
        raise ValueError("invalid defensive mixture")
    result = (1.0 - epsilon) * adaptive + epsilon * reference
    if np.any(result <= 0) or not np.isclose(result.sum(), 1.0):
        raise ValueError("defensive mixture lacks positive support")
    return result


def challenge_policy(challenge: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Turn a surrogate Q-vector into the paper's M=Q*phi action policy.

    The original AdaTE implementation constructs maneuver criticality from
    challenge times the naturalistic action mass and falls back to that mass
    where no challenge is available.  This function is the finite-action,
    highway-env counterpart of that operation.
    """
    challenge, reference = np.asarray(challenge, dtype=float), np.asarray(reference, dtype=float)
    if challenge.shape != reference.shape or np.any(challenge < 0) or np.any(
            reference < 0) or not np.isclose(reference.sum(), 1.0):
        raise ValueError("invalid challenge/reference vectors")
    criticality = challenge * reference
    total = float(criticality.sum())
    return reference.copy() if total <= 1e-15 else criticality / total


def mixture_of_policies(challenges: np.ndarray, alpha: np.ndarray,
                        reference: np.ndarray) -> np.ndarray:
    """Compute psi_alpha = sum_j alpha_j psi_j with one policy per surrogate."""
    challenges, alpha, reference = np.asarray(challenges, dtype=float), np.asarray(
        alpha, dtype=float), np.asarray(reference, dtype=float)
    if challenges.ndim != 2 or challenges.shape[0] != reference.size or challenges.shape[
            1] != alpha.size:
        raise ValueError("challenges must be [actions, surrogates]")
    if np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0):
        raise ValueError("alpha must lie on the simplex")
    policies = np.column_stack(
        [challenge_policy(challenges[:, index], reference) for index in range(alpha.size)])
    mixed = policies @ alpha
    if np.any(mixed < 0) or not np.isclose(mixed.sum(), 1.0):
        raise RuntimeError("invalid mixture of surrogate policies")
    return mixed


def log_importance_weight(reference_probs: np.ndarray, proposal_probs: np.ndarray) -> float:
    reference_probs, proposal_probs = np.asarray(reference_probs,
                                                 dtype=float), np.asarray(proposal_probs,
                                                                          dtype=float)
    if reference_probs.shape != proposal_probs.shape or np.any(reference_probs < 0) or np.any(
            proposal_probs <= 0):
        raise ValueError("invalid probability ledger")
    positive = reference_probs > 0
    return float(np.log(reference_probs[positive]).sum() - np.log(proposal_probs[positive]).sum())


def weighted_event_estimate(events: np.ndarray, log_weights: np.ndarray) -> dict[str, float]:
    events, weights = np.asarray(events, dtype=float), np.exp(np.asarray(log_weights, dtype=float))
    samples = events * weights
    estimate = float(samples.mean()) if len(samples) else float("nan")
    variance = float(samples.var(ddof=1)) if len(samples) > 1 else float("nan")
    half_width = float(1.96 * np.sqrt(
        variance / len(samples))) if len(samples) > 1 and np.isfinite(variance) else float("nan")
    rhw = float(
        half_width /
        abs(estimate)) if np.isfinite(half_width) and abs(estimate) > 1e-15 else float("inf")
    return {
        "estimate": estimate,
        "variance": variance,
        "ci95_half_width": half_width,
        "relative_half_width": rhw,
        "mean_weight": float(weights.mean()) if len(weights) else float("nan"),
        "max_weight": float(weights.max(initial=0.0)),
        "ess": float(weights.sum()**2 / np.square(weights).sum()) if len(weights) else 0.0,
        "events": int(events.sum()),
    }
