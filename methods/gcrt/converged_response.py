"""Converged conditional predictions for separate historical Gaussian priors."""

import torch

from methods.common.classification import torch_moments
from methods.common.gp_laplace import laplace


def component_probabilities(means, covariances, paid, labels):
    if (means.ndim != 2 or covariances.shape != (*means.shape, means.shape[-1])
            or len(paid) != len(labels) or len(set(paid)) != len(paid)
            or any(not 0 <= i < means.shape[1] for i in paid)):
        raise ValueError('Conditional component means, covariances and unique paid labels must align')
    probabilities, diagnostics = [], []
    for mean, covariance in zip(means, covariances):
        if len(paid):
            state = laplace(mean[paid], covariance[paid][:, paid], labels)
            cross = covariance[:, paid]
            predicted_mean = mean+cross @ state.alpha
            solved = torch.linalg.solve_triangular(state.factor, state.root[:, None]*cross.T, upper=False)
            variance = covariance.diagonal()-solved.square().sum(0)
            info = {'iterations': state.iterations, 'gradient_max': state.gradient_max,
                    'backtracks': state.backtracks}
        else:
            predicted_mean, variance = mean, covariance.diagonal()
            info = {'iterations': 0, 'gradient_max': 0., 'backtracks': 0}
        probabilities.append(torch_moments(predicted_mean, variance.clamp_min(1e-10))[0])
        diagnostics.append(info)
    return torch.stack(probabilities), diagnostics
