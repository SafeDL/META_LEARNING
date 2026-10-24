"""Match the initial marginal stage risk under an unchanged latent GP kernel."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
import torch

from methods.common.classification import logistic_moments
from .phase_response import phase_bases


def calibrated_phase_bases(pretrained, history, observations, public_x):
    bases = phase_bases(pretrained, history, observations)
    checks = []
    for phase, base in enumerate(bases):
        registered_mean = float(base.mean.detach())
        frequency = float(expit(registered_mean))
        with torch.no_grad():
            variance = base.covariance(public_x).diagonal().cpu().numpy()

        def probability(mean):
            return float(logistic_moments(np.full(len(variance), mean), variance)[0].mean())

        matched = brentq(lambda mean: probability(mean)-frequency, -10000., 10000., xtol=1e-10)
        checks.append({'phase': phase, 'observed_history_hazard_frequency': frequency,
            'registered_latent_mean': registered_mean,
            'prior_variance_min': float(variance.min()), 'prior_variance_max': float(variance.max()),
            'registered_marginal_probability_mean': probability(registered_mean),
            'marginally_matched_latent_mean': float(matched), 'matched_probability_mean': probability(matched),
            'root_error': abs(probability(matched)-frequency)})
        with torch.no_grad():
            base.mean.fill_(matched)
    return bases, checks
