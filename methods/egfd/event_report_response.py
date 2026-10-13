"""Gaussian event reports conditioned only on observed collision signs."""
import math

import numpy as np
import torch

from .censored_margin_response import CensoredMarginResponse, margin_correlation


def report_constraints(records):
    observed = sorted(q["index"] for q in records)
    keys = [(0, q["index"], -1 if q["collision"] else 1) for q in records]
    keys.sort()
    columns = {index: column for column, index in enumerate(observed)}
    design = np.zeros((len(keys), len(observed)))
    for row, (_, index, sign) in enumerate(keys):
        design[row, columns[index]] = sign
    return observed, keys, design


@torch.no_grad()
def linear_constraint_ep(prior_mean, prior_covariance, observed, design, precision, natural):
    """Parallel damped EP for hard positive linear projections of a Gaussian."""
    kernel = prior_covariance[observed[:, None], observed[None, :]]
    root = torch.linalg.cholesky(kernel)
    projected = design @ root
    base = design @ prior_mean[observed]
    identity = torch.eye(len(observed), dtype=root.dtype, device=root.device)

    def posterior():
        factor = torch.linalg.cholesky(identity + projected.T @ (precision[:, None] * projected))
        mean = torch.cholesky_solve((projected.T @ (natural - precision * base))[:, None], factor).ravel()
        solved = torch.linalg.solve_triangular(factor, projected.T, upper=False)
        return factor, mean, base + projected @ mean, (solved ** 2).sum(dim=0)

    for iteration in range(200):
        factor, mean, marginal_mean, marginal_variance = posterior()
        cavity_precision = 1 / marginal_variance - precision
        if bool((cavity_precision <= 0).any()):
            raise FloatingPointError("Linear EP cavity has nonpositive precision")
        cavity_variance = 1 / cavity_precision
        cavity_mean = cavity_variance * (marginal_mean / marginal_variance - natural)
        z = cavity_mean / torch.sqrt(cavity_variance)
        ratio = torch.exp(-.5 * z ** 2 - .5 * math.log(2 * math.pi) - torch.special.log_ndtr(z))
        strength = (ratio * (ratio + z)).clamp(0, 1)
        tilted_mean = cavity_mean + torch.sqrt(cavity_variance) * ratio
        tilted_variance = cavity_variance * (1 - strength)
        if bool((tilted_variance <= 0).any()):
            raise FloatingPointError("Linear EP tilted variance is nonpositive")
        target_precision = (1 / tilted_variance - cavity_precision).clamp_min(0)
        target_natural = tilted_mean / tilted_variance - cavity_mean * cavity_precision
        delta_precision = .5 * (target_precision - precision)
        delta_natural = .5 * (target_natural - natural)
        change = float(torch.maximum((delta_precision.abs() / (1 + precision.abs())).max(),
                                      (delta_natural.abs() / (1 + natural.abs())).max()))
        precision += delta_precision
        natural += delta_natural
        if change < 1e-7:
            break
    else:
        raise RuntimeError(f"Linear EP sites did not converge: change={change}")
    factor, mean, _, _ = posterior()
    cross = prior_covariance[:, observed]
    whitened = torch.linalg.solve_triangular(root, cross.T, upper=False)
    solved = torch.linalg.solve_triangular(factor, whitened, upper=False)
    mean = prior_mean + whitened.T @ mean
    covariance = prior_covariance - whitened.T @ whitened + solved.T @ solved
    if covariance.diag().min() < -1e-8:
        raise FloatingPointError("Linear EP posterior has negative report variance")
    covariance.diagonal().clamp_(min=0)
    return mean, covariance, precision, natural, iteration + 1, change


class EventReportResponse(CensoredMarginResponse):
    def __init__(self, reference, family, prior, *, device="cuda"):
        self.reference = reference
        self.family = np.asarray(family, dtype=int)
        self.continuous_safe = False
        correlation = margin_correlation(reference, prior)
        original = reference.session.gp
        fraction = original.noise / (np.diag(original.covariance) + original.noise)
        deviation = np.sqrt(1 - fraction)
        report_covariance = correlation * deviation[:, None] * deviation[None, :] + np.diag(fraction)
        self.prior_mean = torch.zeros(len(family), dtype=torch.float64, device=device)
        self.prior_covariance = torch.as_tensor(report_covariance, dtype=torch.float64, device=device)
        self.mean, self.covariance = self.prior_mean.clone(), self.prior_covariance.clone()
        # Independent report noise is already inside the report covariance.
        self.noise = torch.zeros_like(self.mean)
        self.indices, self.records, self.selection_checks, self.ep_updates = [], [], [], []
        self.sites = {}

    @torch.no_grad()
    def condition(self, observations):
        self.indices = [int(q["index"]) for q in observations]
        self.records = [dict(q) for q in observations]
        self.sites = {}
        self.ep_updates = []
        self.refit()

    @torch.no_grad()
    def observe(self, index, risk, collision):
        if index in self.indices or not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid or duplicate event feedback")
        self.records.append({"index": int(index), "risk": float(risk), "collision": bool(collision),
                             "predictive_mean_before": float(self.mean[index]),
                             "predictive_variance_before": float(self.covariance[index, index]),
                             "likelihood": "event_sign", "observed_proxy": None})
        self.indices.append(int(index))
        self.refit()

    @torch.no_grad()
    def refit(self):
        observed, keys, design = report_constraints(self.records)
        device, dtype = self.mean.device, self.mean.dtype
        values = np.asarray([self.sites.get(key, (0., 0.)) for key in keys])
        precision = torch.as_tensor(values[:, 0], dtype=dtype, device=device)
        natural = torch.as_tensor(values[:, 1], dtype=dtype, device=device)
        result = linear_constraint_ep(self.prior_mean, self.prior_covariance,
                                      torch.as_tensor(observed, dtype=torch.long, device=device),
                                      torch.as_tensor(design, dtype=dtype, device=device), precision, natural)
        self.mean, self.covariance, precision, natural, iterations, change = result
        values = torch.stack((precision, natural), dim=1).cpu().numpy()
        self.sites = {key: tuple(value) for key, value in zip(keys, values)}
        self.ep_updates.append({"observations": len(observed), "constraints": len(keys),
                                "iterations": iterations, "relative_site_change": change})

    def diagnostics(self):
        return {**super().diagnostics(),
                "ep_updates": self.ep_updates,
                "inference": "Gaussian EP of event signs; report covariance includes white noise"}
