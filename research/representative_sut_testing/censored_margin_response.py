"""Continuous safe proxy observations and censored failure events."""
import math

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr, ndtri
import torch

from methods.history_guided_testing.config import ROOT as HISTORY_ROOT
from methods.history_guided_testing.history import split_indices

from .config import PROFILES


def nearest_average(x, source_x, values):
    distance = cdist(x[:, :4], source_x[:, :4])
    distance[x[:, 4, None] != source_x[None, :, 4]] = np.inf
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :8]
    selected = np.take_along_axis(distance, nearest, axis=1)
    if not np.isfinite(selected).all():
        raise ValueError("Historical margin prior requires eight observations per functional family")
    weight = np.exp(-.5 * (selected / .15) ** 2)
    weight /= weight.sum(axis=1, keepdims=True)
    return (weight * values[nearest]).sum(axis=1)


def calibrated_margin_moments(probability, positive_mean):
    standardized_mean = -ndtri(probability)
    density = np.exp(-.5 * standardized_mean ** 2) / math.sqrt(2 * math.pi)
    deviation = positive_mean / (standardized_mean + density / ndtr(standardized_mean))
    return standardized_mean * deviation, deviation ** 2


def historical_margin_prior(x):
    events, positive_means, counts = [], [], {}
    with np.load(HISTORY_ROOT / "history/responses.npz") as bank:
        train, _ = split_indices(bank["x"])
        source_x = bank["x"][train]
        for profile in PROFILES:
            event = bank[profile.name + "_collision"][train].astype(bool)
            risk = bank[profile.name][train]
            events.append(nearest_average(x, source_x, event.astype(float)))
            positive_means.append(nearest_average(x, source_x[~event], 1 - risk[~event]))
            counts[profile.name] = {str(g): int(((source_x[:, 4] == g) & ~event).sum()) for g in (0, 1)}
    events = np.column_stack(events)
    probability = np.clip(events.mean(axis=1), .01, .99)
    positive_mean = np.column_stack(positive_means).mean(axis=1)
    if np.any(positive_mean <= 0):
        raise ValueError("Historical positive margin must be strictly positive")
    mean, variance = calibrated_margin_moments(probability, positive_mean)
    scores = -ndtri(np.clip(events, .01, .99))
    modes = (scores - scores.mean(axis=1, keepdims=True)) / math.sqrt(len(PROFILES) - 1)
    return {"mean": mean, "report_variance": variance, "modes": modes,
            "probability": probability, "positive_mean": positive_mean,
            "safe_source_counts": counts,
            "scope": "original six sources and frozen spatial training indices; no target labels"}


def neutral_margin_prior(historical):
    probability = np.full_like(historical["probability"], .5)
    mean, variance = calibrated_margin_moments(probability, historical["positive_mean"])
    return {**historical, "probability": probability, "mean": mean, "report_variance": variance,
            "scope": "fixed neutral event probability; same source safe-proxy mean and response correlation modes"}


def margin_correlation(reference, prior):
    original = reference.session.gp
    deviation = np.sqrt(np.diag(original.covariance))
    correlation = original.covariance / (deviation[:, None] * deviation[None, :])
    correlation += prior["modes"] @ prior["modes"].T
    normalization = np.sqrt(np.diag(correlation))
    return correlation / (normalization[:, None] * normalization[None, :])


class CensoredMarginResponse:
    def __init__(self, reference, family, prior, *, continuous_safe, device="cuda"):
        self.reference = reference
        self.family = np.asarray(family, dtype=int)
        self.continuous_safe = continuous_safe
        original = reference.session.gp
        correlation = margin_correlation(reference, prior)
        fraction = original.noise / (np.diag(original.covariance) + original.noise)
        signal_sd = np.sqrt(prior["report_variance"] * (1 - fraction))
        covariance = correlation * signal_sd[:, None] * signal_sd[None, :]
        self.mean = torch.as_tensor(prior["mean"].copy(), dtype=torch.float64, device=device)
        self.covariance = torch.as_tensor(covariance, dtype=torch.float64, device=device)
        self.noise = torch.as_tensor(prior["report_variance"] * fraction, dtype=torch.float64, device=device)
        self.indices, self.records, self.selection_checks = [], [], []

    @torch.no_grad()
    def probabilities(self):
        return torch.special.ndtr(-self.mean / torch.sqrt(self.covariance.diag() + self.noise)).cpu().numpy()

    @torch.no_grad()
    def observe(self, index, risk, collision):
        if index in self.indices or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid or duplicate censored-margin feedback")
        cross = self.covariance[:, index].clone()
        mean = self.mean[index].clone()
        variance = self.covariance[index, index] + self.noise[index]
        sd = torch.sqrt(variance)
        if not collision and self.continuous_safe:
            value = 1 - risk
            innovation = (value - mean) / sd
            strength = torch.ones_like(mean)
            likelihood = "positive_value"
        else:
            value = None
            sign = -1 if collision else 1
            z = sign * mean / sd
            ratio = torch.exp(-.5 * z ** 2 - .5 * math.log(2 * math.pi) - torch.special.log_ndtr(z))
            innovation = sign * ratio
            strength = ratio * (ratio + z)
            if not -1e-8 <= float(strength) <= 1 + 1e-8:
                raise FloatingPointError("Invalid truncated-normal variance factor")
            strength = strength.clamp(0, 1)
            likelihood = "negative_censor" if collision else "positive_censor"
        self.mean += cross * (innovation / sd)
        self.covariance -= torch.outer(cross, cross) * (strength / variance)
        if self.covariance.diag().min() < -1e-8:
            raise FloatingPointError("Censored-margin update produced negative latent variance")
        self.covariance.diagonal().clamp_(min=0)
        self.indices.append(int(index))
        self.records.append({"index": int(index), "risk": float(risk), "collision": bool(collision),
                             "likelihood": likelihood, "observed_proxy": value,
                             "predictive_mean_before": float(mean), "predictive_variance_before": float(variance),
                             "standardized_mean_innovation": float(innovation),
                             "conditional_variance_factor": float(strength)})

    def rank(self, remaining, *, tie_break):
        head = int(self.reference.head())
        indices = np.flatnonzero(remaining & (self.family == self.family[head]))
        probability = self.probabilities()
        mean = self.mean.cpu().numpy()
        order = np.lexsort((tie_break[indices], mean[indices], -np.round(probability[indices], 12)))
        proposed = int(indices[order[0]])
        chosen = proposed if probability[proposed] > probability[head] + 1e-12 else head
        self.selection_checks.append({"query": len(self.indices) + 1, "chosen_index": chosen,
                                      "reference_index": head, "family": int(self.family[head]),
                                      "chosen_probability": float(probability[chosen]),
                                      "reference_probability": float(probability[head])})
        return np.asarray([chosen])

    def diagnostics(self):
        return {"margin_records": self.records, "margin_observations": len(self.indices),
                "regime_selection_checks": self.selection_checks,
                "inference": "Gaussian assumed-density filtering; inequality moments for failures, numeric positive proxy for safe mixed observations",
                "continuous_safe_feedback": self.continuous_safe}
