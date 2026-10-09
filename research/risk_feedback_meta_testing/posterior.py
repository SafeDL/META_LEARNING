"""Use the same bounded risk observation model in training and deployment."""
import math

import numpy as np
import torch

from .risk_state import CensoredRiskState, risk_kernel


def condition_gaussian_risk(x, prediction, support, feedback, discrepancy):
    covariance, noise = risk_kernel(x, discrepancy)
    weights = prediction.new_full((len(prediction),), -math.log(len(prediction)))
    if len(support) == 0:
        return weights, torch.zeros_like(prediction), covariance.diagonal()
    support = torch.as_tensor(support, dtype=torch.long, device=x.device)
    feedback = torch.as_tensor(feedback, dtype=x.dtype, device=x.device)
    observed_covariance = (covariance[support][:, support]
                           + torch.diag(noise[support]))
    factor = torch.linalg.cholesky(observed_covariance)
    residual = feedback[None, :] - prediction[:, support]
    solved = torch.cholesky_solve(residual.T, factor)
    weights = weights - 0.5 * (residual * solved.T).sum(1)
    weights = weights - torch.logsumexp(weights, 0)
    cross = covariance[:, support]
    mean = (cross @ solved).T
    reduction = (cross * torch.cholesky_solve(cross.T, factor).T).sum(1)
    variance = (covariance.diagonal() - reduction).clamp_min(0)
    return weights, mean, variance


def condition_risk(x, prediction, support, feedback, discrepancy):
    """Exact Gaussian conditioning inside bounds; moment filtering at atoms."""
    if len(support) == 0 or np.all((np.asarray(feedback) > 0)
                                 & (np.asarray(feedback) < 1)):
        weights, mean, variance = condition_gaussian_risk(
            x, prediction, support, feedback, discrepancy)
        return weights, mean, variance[None, :].expand_as(prediction)
    state = CensoredRiskState(x, prediction, discrepancy, len(support))
    for index, risk in zip(support, feedback):
        state.observe(int(index), float(risk))
    return state.log_weights, state.mean - state.baseline, state.variance
