"""Batch independent virtual observations without changing Laplace conditioning."""

import torch
from torch.nn import functional as F

from methods.common.classification import torch_moments
from methods.common.gp_laplace import MODE_ITERATIONS, MODE_TOLERANCE


def _modes(mean, covariance, labels):
    size = mean.shape[-1]
    eye = torch.eye(size, dtype=mean.dtype, device=mean.device)
    covariance = covariance + 1e-5 * eye
    prior_factor = torch.linalg.cholesky(covariance)
    transpose = prior_factor.transpose(-1, -2)
    whitened = torch.zeros_like(mean)
    iterations = torch.zeros(len(mean), dtype=torch.int16, device=mean.device)
    backtracks = torch.zeros_like(iterations)
    for _ in range(MODE_ITERATIONS):
        latent = mean + (prior_factor @ whitened[..., None])[..., 0]
        probability = latent.sigmoid()
        gradient = whitened + (transpose @ (probability - labels)[..., None])[..., 0]
        active = gradient.abs().amax(-1) > MODE_TOLERANCE
        if not bool(active.any()):
            break
        weight = (probability * (1 - probability)).clamp_min(1e-12)
        hessian = eye + transpose @ (weight[..., None] * prior_factor)
        factor = torch.linalg.cholesky(hessian)
        direction = -torch.cholesky_solve(gradient[..., None], factor)[..., 0]
        direction = torch.where(active[:, None], direction, 0.)
        slope = (gradient * direction).sum(-1)
        near = active & (-slope < 1e-12)
        objective = (.5 * whitened.square().sum(-1)
                     + F.binary_cross_entropy_with_logits(latent, labels, reduction='none').sum(-1))
        whitened = whitened + torch.where(near[:, None], direction, 0.)
        pending = active & ~near
        rate = torch.ones_like(slope)
        for _ in range(30):
            if not bool(pending.any()):
                break
            candidate = whitened + rate[:, None] * direction
            latent_candidate = mean + (prior_factor @ candidate[..., None])[..., 0]
            value = (.5 * candidate.square().sum(-1)
                     + F.binary_cross_entropy_with_logits(latent_candidate, labels, reduction='none').sum(-1))
            accepted = pending & (value <= objective + 1e-4 * rate * slope)
            whitened = torch.where(accepted[:, None], candidate, whitened)
            pending = pending & ~accepted
            backtracks += pending.to(backtracks.dtype)
            rate = torch.where(pending, rate * .5, rate)
        if bool(pending.any()):
            raise FloatingPointError('Batched Bernoulli line search did not find a decreasing step')
        iterations += active.to(iterations.dtype)
    latent = mean + (prior_factor @ whitened[..., None])[..., 0]
    probability = latent.sigmoid()
    gradient = whitened + (transpose @ (probability - labels)[..., None])[..., 0]
    residual = gradient.abs().amax(-1)
    if bool((residual > MODE_TOLERANCE).any()):
        raise FloatingPointError(f'Batched Bernoulli mode did not converge: gradient {float(residual.max()):.3g}')
    root = (probability * (1 - probability)).clamp_min(1e-12).sqrt()
    factor = torch.linalg.cholesky(eye + root[..., :, None] * covariance * root[..., None, :])
    alpha = torch.linalg.solve_triangular(transpose, whitened[..., None], upper=True)[..., 0]
    return alpha, root, factor, {'gradients': residual, 'iterations': iterations, 'backtracks': backtracks}


@torch.no_grad()
def virtual_probabilities(means, covariances, paid, labels, candidates, *, chunk_size=32):
    """Return candidate × Boolean × component × pool probabilities and mode diagnostics."""
    if (means.ndim != 2 or covariances.shape != (*means.shape, means.shape[-1])
            or means.dtype != torch.float64 or covariances.dtype != means.dtype
            or means.device != covariances.device or len(paid) != len(labels)
            or len(set(paid)) != len(paid) or len(set(candidates)) != len(candidates)
            or set(paid) & set(candidates) or chunk_size < 1
            or any(not 0 <= i < means.shape[-1] for i in [*paid, *candidates])):
        raise ValueError('Virtual batches require aligned float64 priors and distinct unpaid candidates')
    component_count, pool_size = means.shape
    output = means.new_empty((len(candidates), 2, component_count, pool_size))
    information = {key: torch.empty((len(candidates), 2, component_count), device=means.device,
                                  dtype=torch.float64 if key == 'gradients' else torch.int16)
                   for key in ('gradients', 'iterations', 'backtracks')}
    observed = torch.as_tensor(paid, dtype=torch.long, device=means.device)
    paid_labels = means.new_tensor(labels)
    count = len(paid) + 1
    for start in range(0, len(candidates), chunk_size):
        current = torch.as_tensor(candidates[start:start + chunk_size], dtype=torch.long, device=means.device)
        width = len(current)
        indices = torch.cat((observed.expand(width, -1), current[:, None]), dim=-1)
        cross = covariances[:, indices].permute(1, 0, 2, 3)
        covariance = cross.gather(-1, indices[:, None, None, :].expand(width, component_count, count, count))
        mean = means[:, indices].permute(1, 0, 2)
        covariance = covariance[:, None].expand(-1, 2, -1, -1, -1).reshape(-1, count, count)
        mean = mean[:, None].expand(-1, 2, -1, -1).reshape(-1, count)
        response = means.new_empty((width, 2, component_count, count))
        response[..., :-1] = paid_labels
        response[:, 0, :, -1] = 0.
        response[:, 1, :, -1] = 1.
        alpha, root, factor, diagnostics = _modes(mean, covariance, response.reshape(-1, count))
        cross = cross[:, None].expand(-1, 2, -1, -1, -1).reshape(-1, count, pool_size)
        predicted_mean = (means[None, None].expand(width, 2, -1, -1).reshape(-1, pool_size)
                          + (cross.transpose(-1, -2) @ alpha[..., None])[..., 0])
        solved = torch.linalg.solve_triangular(factor, root[..., None] * cross, upper=False)
        variance = (covariances.diagonal(dim1=-2, dim2=-1)[None, None]
                    .expand(width, 2, -1, -1).reshape(-1, pool_size) - solved.square().sum(-2))
        probability = torch_moments(predicted_mean, variance.clamp_min(1e-10))[0]
        output[start:start + width] = probability.reshape(width, 2, component_count, pool_size)
        for key, value in diagnostics.items():
            information[key][start:start + width] = value.reshape(width, 2, component_count)
    return output, information
