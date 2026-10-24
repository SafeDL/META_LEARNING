"""Collision-only adaptation under fixed historical Gaussian response priors."""

import torch

from methods.common.classification import torch_moments
from .joint_response import logistic_laplace


def conditional_prior(model, x, source_means=None, source_covariances=None, reference_mean=None):
    mean = model.mean.expand(len(x))
    covariance = model.covariance(x)
    if source_means is None:
        return mean[None], covariance[None]
    means = source_means + (model.mean-reference_mean)
    covariances = source_covariances + covariance[None]
    return torch.cat((means, mean[None])), torch.cat((covariances, covariance[None]))


def fit_conditional(model, x, labels, weights, *, source_means=None, source_covariances=None,
                    reference_mean=None, steps=10, learning_rate=.003):
    """Minimize a fixed weighted sum of component Laplace negative evidences."""
    weights = torch.as_tensor(weights, dtype=torch.float64, device=model.mean.device).detach()
    expected = 1 if source_means is None else len(source_means)+1
    if weights.shape != (expected,) or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.)):
        raise ValueError('Every conditional component needs one normalized nonnegative weight')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mean, covariance = conditional_prior(model, x, source_means, source_covariances, reference_mean)
        component_loss = logistic_laplace(mean, covariance, labels)[0]
        loss = (weights*component_loss).sum()/len(labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite conditional collision GP evidence')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def component_probabilities(mean, covariance, paid, labels):
    if len(paid):
        _, alpha, root, factor = logistic_laplace(mean[..., paid], covariance[..., paid, :][..., :, paid], labels)
        cross = covariance[..., :, paid]
        mean = mean+(cross @ alpha[..., None])[..., 0]
        solved = torch.linalg.solve_triangular(factor, root[..., :, None]*cross.transpose(-1, -2), upper=False)
        variance = covariance.diagonal(dim1=-2, dim2=-1)-solved.square().sum(-2)
    else:
        variance = covariance.diagonal(dim1=-2, dim2=-1)
    return torch_moments(mean, variance.clamp_min(1e-10))[0]
