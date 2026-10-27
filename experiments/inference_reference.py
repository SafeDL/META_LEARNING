"""Independent NumPy inference used by collision acquisition audits."""

import numpy as np
from scipy.optimize import root
from scipy.special import expit


def mode_numpy(mean, covariance, labels):
    labels = np.asarray(labels, dtype=float)
    k = covariance + 1e-5 * np.eye(len(labels))
    factor = np.linalg.cholesky(k)

    def gradient(z):
        return z + factor.T @ (expit(mean + factor @ z) - labels)

    def hessian(z):
        p = expit(mean + factor @ z)
        return np.eye(len(labels)) + factor.T @ ((p * (1 - p))[:, None] * factor)

    solved = root(
        gradient,
        np.zeros(len(labels)),
        jac=hessian,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 1000},
    )
    maximum = float(np.max(np.abs(gradient(solved.x))))
    if maximum > 5e-9:
        raise FloatingPointError(
            f"Independent Bernoulli root failed: gradient {maximum}; {solved.message}"
        )
    latent = mean + factor @ solved.x
    p = expit(latent)
    curvature = np.sqrt(np.maximum(p * (1 - p), 1e-12))
    posterior_factor = np.linalg.cholesky(
        np.eye(len(labels)) + curvature[:, None] * k * curvature[None, :]
    )
    nll = (
        0.5 * np.square(solved.x).sum()
        + np.sum(np.logaddexp(0, latent) - labels * latent)
        + np.log(np.diag(posterior_factor)).sum()
    )
    return nll, np.linalg.solve(factor.T, solved.x), curvature, posterior_factor, maximum


def posterior_numpy(mean, covariance, observed, labels):
    indices = np.asarray(observed, dtype=int)
    state = mode_numpy(mean[indices], covariance[np.ix_(indices, indices)], labels)
    _, alpha, curvature, factor, _ = state
    cross = covariance[indices]
    solved = np.linalg.solve(factor, curvature[:, None] * cross)
    return mean + cross.T @ alpha, covariance - solved.T @ solved


def probabilities_numpy(means, covariances, paid, labels):
    from scipy.special import expit

    nodes, weights = np.polynomial.hermite.hermgauss(32)
    probabilities = []
    for mean, covariance in zip(means, covariances):
        if len(paid):
            _, alpha, curvature, factor, _ = mode_numpy(
                mean[paid], covariance[np.ix_(paid, paid)], labels
            )
            cross = covariance[np.asarray(paid)]
            solved = np.linalg.solve(factor, curvature[:, None] * cross)
            m = mean + cross.T @ alpha
            variance = np.diag(covariance) - np.square(solved).sum(0)
        else:
            m, variance = mean, np.diag(covariance)
        probabilities.append(
            expit(m[:, None] + np.sqrt(2 * np.maximum(variance, 1e-10))[:, None] * nodes)
            @ (weights / np.sqrt(np.pi))
        )
    return np.asarray(probabilities)
