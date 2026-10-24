"""Joint Gaussian/Laplace predictions, including fixed-kernel classification folds."""

import torch


def bernoulli_posterior(mean, covariance, observed, labels, *, iterations=10):
    """Condition a latent GP on Bernoulli labels; folds may have different supports.

    The final dimension indexes a common prediction pool. A leading dimension
    in ``observed`` and ``labels`` requests independently fitted Laplace modes,
    rather than removing a label from an already fitted Gaussian approximation.
    """
    observed = torch.as_tensor(observed, dtype=torch.long, device=mean.device)
    labels = torch.as_tensor(labels, dtype=mean.dtype, device=mean.device)
    count = mean.shape[-1]
    batch = torch.broadcast_shapes(mean.shape[:-1], covariance.shape[:-2],
                                   observed.shape[:-1], labels.shape[:-1])
    mean = mean.expand(*batch, count)
    covariance = covariance.expand(*batch, count, count)
    support_count = observed.shape[-1]
    if not support_count:
        return mean, covariance
    observed = observed.expand(*batch, support_count)
    labels = labels.expand(*batch, support_count)
    support_mean = mean.gather(-1, observed)
    cross = covariance.gather(-2, observed.unsqueeze(-1).expand(*batch, support_count, count))
    support = cross.gather(-1, observed.unsqueeze(-2).expand(*batch, support_count, support_count))
    eye = torch.eye(support_count, dtype=mean.dtype, device=mean.device)
    support = support + 1e-5 * eye
    latent = support_mean
    for _ in range(iterations):
        probability = latent.sigmoid()
        weight = (probability * (1 - probability)).clamp_min(1e-9)
        root = weight.sqrt()
        factor = torch.linalg.cholesky(eye + root.unsqueeze(-1) * support * root.unsqueeze(-2))
        b = weight * (latent - support_mean) + labels - probability
        rhs = root * (support @ b.unsqueeze(-1)).squeeze(-1)
        alpha = b - root * torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
        latent = support_mean + (support @ alpha.unsqueeze(-1)).squeeze(-1)
    probability = latent.sigmoid()
    root = (probability * (1 - probability)).clamp_min(1e-9).sqrt()
    factor = torch.linalg.cholesky(eye + root.unsqueeze(-1) * support * root.unsqueeze(-2))
    alpha = torch.cholesky_solve((latent - support_mean).unsqueeze(-1),
                                 torch.linalg.cholesky(support)).squeeze(-1)
    posterior_mean = mean + (cross.transpose(-1, -2) @ alpha.unsqueeze(-1)).squeeze(-1)
    solved = torch.linalg.solve_triangular(factor, root.unsqueeze(-1) * cross, upper=False)
    posterior_covariance = covariance - solved.transpose(-1, -2) @ solved
    posterior_covariance = .5 * (posterior_covariance + posterior_covariance.transpose(-1, -2))
    return posterior_mean, posterior_covariance
