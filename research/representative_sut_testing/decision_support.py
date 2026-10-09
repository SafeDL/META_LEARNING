"""Posterior support for an event-propensity ordering, retaining covariance."""
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from .config import BUDGET


SUPPORT_THRESHOLD = 1 - 1 / np.sqrt(BUDGET)




def bivariate_normal_probability(first, second, correlation):
    """Deterministic Plackett integral for the standard bivariate normal CDF."""
    if abs(correlation) >= 1:
        raise FloatingPointError("Report correlation must be nonsingular with positive observation noise")
    independent = float(ndtr(first) * ndtr(second))
    if correlation == 0:
        return independent

    def density(value):
        denominator = 1 - value ** 2
        exponent = -(first ** 2 - 2 * value * first * second + second ** 2) / (2 * denominator)
        return np.exp(exponent) / (2 * np.pi * np.sqrt(denominator))

    correction = quad(density, 0, correlation, epsabs=1e-11, epsrel=1e-9)[0]
    return float(np.clip(independent + correction, 0, 1))


def discovery_loss_support(candidate, proposed, reference):
    """Risk of a lost immediate binary discovery, rather than strict ordering."""
    weights = candidate.selection_weights()
    components = []
    for model in candidate.models:
        first = float(model.covariance[proposed, proposed] + model.noise[proposed])
        second = float(model.covariance[reference, reference] + model.noise[reference])
        means = [float(model.mean[proposed]), float(model.mean[reference])]
        sd = np.sqrt([first, second])
        correlation = float(model.covariance[proposed, reference]) / (sd[0] * sd[1])
        loss = bivariate_normal_probability(means[0] / sd[0], -means[1] / sd[1], -correlation)
        gain = bivariate_normal_probability(-means[0] / sd[0], means[1] / sd[1], -correlation)
        probabilities = ndtr(-np.asarray(means) / sd)
        if abs((gain - loss) - (probabilities[0] - probabilities[1])) > 1e-8:
            raise FloatingPointError("Joint binary gain/loss does not match its marginal probabilities")
        components.append({"loss_probability": loss, "gain_probability": gain,
                           "candidate_probability": float(probabilities[0]), "reference_probability": float(probabilities[1]),
                           "report_correlation": correlation, "report_standardized_means": (np.asarray(means) / sd).tolist()})
    loss = float(weights @ np.asarray([c["loss_probability"] for c in components]))
    gain = float(weights @ np.asarray([c["gain_probability"] for c in components]))
    return {"proposed_index": int(proposed), "reference_index": int(reference), "model_weights": weights.tolist(),
            "components": components, "loss_probability": loss, "gain_probability": gain,
            "expected_immediate_gain": gain - loss, "loss_allowance": float(1 - SUPPORT_THRESHOLD),
            "supported": bool(proposed != reference and loss <= 1 - SUPPORT_THRESHOLD and gain - loss > 1e-12),
            "scope": "EP working probability of candidate miss and reference hit; no true-world confidence or full-curve claim"}
