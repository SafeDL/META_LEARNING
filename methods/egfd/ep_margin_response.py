"""Joint EP refitting of Gaussian safe reports and censored collision sites."""
import math

import torch

from .censored_margin_response import CensoredMarginResponse


class ExpectationPropagatedMargin(CensoredMarginResponse):
    def __init__(self, reference, family, prior, *, continuous_safe, device="cuda"):
        super().__init__(reference, family, prior, continuous_safe=continuous_safe, device=device)
        self.initialize_sites()

    def initialize_sites(self):
        self.prior_mean = self.mean.clone()
        self.prior_covariance = self.covariance.clone()
        self.site_precision = torch.zeros_like(self.mean)
        self.site_natural_mean = torch.zeros_like(self.mean)
        self.ep_updates = []

    def observed_posterior(self, observed, precision, natural):
        kernel = self.prior_covariance[observed[:, None], observed[None, :]]
        root = torch.sqrt(precision)
        factor = torch.linalg.cholesky(torch.eye(len(observed), dtype=kernel.dtype, device=kernel.device)
                                      + root[:, None] * kernel * root[None, :])
        solved = torch.linalg.solve_triangular(factor, root[:, None] * kernel, upper=False)
        covariance = kernel - solved.T @ solved
        residual = natural - precision * self.prior_mean[observed]
        alpha = residual - root * torch.cholesky_solve((root * (kernel @ residual))[:, None], factor).ravel()
        mean = self.prior_mean[observed] + kernel @ alpha
        return mean, covariance, factor, alpha

    @torch.no_grad()
    def observe(self, index, risk, collision):
        # This records the before-query likelihood and the single-site moments.
        # The inference state below is then jointly refitted using every site.
        super().observe(index, risk, collision)
        if not collision and self.continuous_safe:
            self.site_precision[index] = 1 / self.noise[index]
            self.site_natural_mean[index] = (1 - risk) / self.noise[index]
        self.refit()

    @torch.no_grad()
    def condition(self, observations):
        """Batch posterior for paid-prefix diagnostics, without prequential logs."""
        self.indices = [int(q["index"]) for q in observations]
        self.records = [dict(q) for q in observations]
        self.site_precision.zero_()
        self.site_natural_mean.zero_()
        self.ep_updates = []
        for q in observations:
            if not q["collision"] and self.continuous_safe:
                self.site_precision[q["index"]] = 1 / self.noise[q["index"]]
                self.site_natural_mean[q["index"]] = (1 - q["risk"]) / self.noise[q["index"]]
        self.refit()

    @torch.no_grad()
    def refit(self):
        observed = torch.as_tensor(sorted(self.indices), dtype=torch.long, device=self.mean.device)
        precision = self.site_precision[observed].clone()
        natural = self.site_natural_mean[observed].clone()
        flags = {r["index"]: r["collision"] or not self.continuous_safe for r in self.records}
        signs = {r["index"]: -1 if r["collision"] else 1 for r in self.records}
        censored = torch.as_tensor([flags[int(i)] for i in observed], dtype=torch.bool, device=self.mean.device)
        sign = torch.as_tensor([signs[int(i)] for i in observed[censored]], dtype=self.mean.dtype, device=self.mean.device)
        noise = self.noise[observed][censored]
        change = 0.
        for iteration in range(200):
            mean, covariance, _, _ = self.observed_posterior(observed, precision, natural)
            if not bool(censored.any()):
                break
            marginal_variance = covariance.diag()[censored]
            cavity_precision = 1 / marginal_variance - precision[censored]
            if bool((cavity_precision <= 0).any()):
                raise FloatingPointError("EP cavity has nonpositive precision")
            cavity_variance = 1 / cavity_precision
            cavity_mean = cavity_variance * (mean[censored] / marginal_variance - natural[censored])
            report_variance = cavity_variance + noise
            z = sign * cavity_mean / torch.sqrt(report_variance)
            ratio = torch.exp(-.5 * z ** 2 - .5 * math.log(2 * math.pi) - torch.special.log_ndtr(z))
            strength = (ratio * (ratio + z)).clamp(0, 1)
            tilted_mean = cavity_mean + sign * cavity_variance / torch.sqrt(report_variance) * ratio
            tilted_variance = cavity_variance - cavity_variance ** 2 / report_variance * strength
            if bool((tilted_variance <= 0).any()):
                raise FloatingPointError("EP tilted variance is nonpositive")
            target_precision = (1 / tilted_variance - cavity_precision).clamp_min(0)
            target_natural = tilted_mean / tilted_variance - cavity_mean * cavity_precision
            delta_precision = .5 * (target_precision - precision[censored])
            delta_natural = .5 * (target_natural - natural[censored])
            change = float(torch.maximum((delta_precision.abs() / (1 + precision[censored].abs())).max(),
                                          (delta_natural.abs() / (1 + natural[censored].abs())).max()))
            precision[censored] += delta_precision
            natural[censored] += delta_natural
            if change < 1e-7:
                break
        else:
            raise RuntimeError(f"EP margin sites did not converge: change={change}")
        _, _, factor, alpha = self.observed_posterior(observed, precision, natural)
        cross = self.prior_covariance[:, observed]
        solved = torch.linalg.solve_triangular(factor, torch.sqrt(precision)[:, None] * cross.T, upper=False)
        self.mean = self.prior_mean + cross @ alpha
        self.covariance = self.prior_covariance - solved.T @ solved
        if self.covariance.diag().min() < -1e-8:
            raise FloatingPointError("EP posterior has negative latent variance")
        self.covariance.diagonal().clamp_(min=0)
        self.site_precision[observed] = precision
        self.site_natural_mean[observed] = natural
        self.ep_updates.append({"index": int(self.indices[-1]), "observations": len(self.indices),
                                "iterations": iteration + 1, "relative_site_change": change})

    def diagnostics(self):
        return {**super().diagnostics(), "inference": "joint damped Gaussian expectation propagation; all observed sites refitted",
                "ep_updates": self.ep_updates}
