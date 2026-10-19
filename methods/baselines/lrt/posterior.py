"""Consistent block Student mixtures with source-specific covariance geometry."""

import numpy as np
from scipy.special import logsumexp, stdtr
from scipy.stats import gamma

from methods.baselines.cg_brt.posterior import RiskPosterior
from methods.baselines.cg_brt.ranking import (
    bivariate_normal_cdf,
)
from methods.baselines.saem.posterior import student_log_density

from .config import JITTER, MODELS, NOISE


class BlockPosterior:
    def __init__(self, means, covariance_groups, probability):
        self.groups = []
        self.count = means.shape[1]
        self.log_weight = np.log(probability)
        self.log_weight -= logsumexp(self.log_weight)
        for columns, covariance in covariance_groups:
            self.groups.append(
                (
                    np.asarray(columns),
                    RiskPosterior(
                        covariance,
                        means[:, columns],
                        np.ones(len(columns)) / len(columns),
                        means[:, columns].mean(1),
                    ),
                )
            )

    @property
    def alpha(self):
        return self.groups[0][1].alpha

    def parameters(self, indices):
        indices = np.asarray(indices)
        mean = np.empty((len(indices), self.count))
        variance = np.empty_like(mean)
        beta = np.empty(self.count)
        for columns, model in self.groups:
            mean[:, columns] = model.source_mean[indices]
            variance[:, columns] = model.variance[indices, None] + NOISE + JITTER
            beta[columns] = model.beta
        return mean, variance, beta

    def probability(self, indices):
        mean, variance, beta = self.parameters(indices)
        z = (mean - 0.5) / np.sqrt(variance * beta / self.alpha)
        return stdtr(2 * self.alpha, z) @ np.exp(self.log_weight)

    def observe(self, index, risk):
        mean, variance, beta = self.parameters([index])
        density = student_log_density(
            risk - mean[0], beta / self.alpha * variance[0], 2 * self.alpha
        )
        for _, model in self.groups:
            model.observe(index, risk)
        self.log_weight += density
        self.log_weight -= logsumexp(self.log_weight)

    def conditional(self, index, values, indices):
        values = np.asarray(values).reshape(-1)
        mean, variance, beta = self.parameters(indices)
        current, denominator, _ = self.parameters([index])
        cross = np.empty_like(mean)
        for columns, model in self.groups:
            cross[:, columns] = (
                model.covariance[indices, index]
                - model.solved_cross[indices] @ model.solved_cross[index]
            )[:, None]
        residual = values[:, None] - current[0]
        updated_mean = mean[None] + cross[None] * residual[:, None] / denominator[0]
        updated_variance = variance - cross**2 / denominator[0]
        updated_beta = beta + 0.5 * residual**2 / denominator[0]
        weight = self.log_weight + student_log_density(
            residual, beta / self.alpha * denominator[0], 2 * self.alpha
        )
        weight -= logsumexp(weight, axis=-1, keepdims=True)
        return updated_mean, updated_variance, self.alpha + 0.5, updated_beta, weight

    def draws(self, indices, count, generator):
        source = generator.choice(self.count, count, p=np.exp(self.log_weight))
        scale_precision = generator.gamma(self.alpha, size=count)
        normal = generator.standard_normal((count, len(indices)))
        output = np.empty_like(normal)
        for columns, model in self.groups:
            factor = np.linalg.cholesky(model.covariance_block(indices))
            correlated = normal @ factor.T
            for offset, column in enumerate(columns):
                rows = source == column
                output[rows] = model.source_mean[indices, offset] + correlated[
                    rows
                ] * np.sqrt(model.beta[offset] / scale_precision[rows, None])
        return output

    def losses(self, indices, reference):
        output = np.zeros(len(indices))
        for columns, model in self.groups:
            block = model.covariance_block(np.r_[indices, reference])
            sd = np.sqrt(np.diag(block))
            correlation = block[:-1, -1] / (sd[:-1] * sd[-1])
            precision = gamma.ppf(model.scale_quantiles, a=model.alpha)
            scales = np.sqrt(model.beta[:, None] / precision)
            za = (model.source_mean[indices] - 0.5) / sd[:-1, None]
            zb = (model.source_mean[reference] - 0.5) / sd[-1]
            event = bivariate_normal_cdf(
                -za[..., None] / scales,
                zb[None, :, None] / scales,
                -correlation[:, None, None],
            )
            output += (event @ model.scale_weights) @ np.exp(self.log_weight[columns])
        output[np.asarray(indices) == reference] = 0.0
        return output


class LocalPosterior:
    def __init__(
        self,
        prior,
        kappa=1.0,
        target_probability=0.25,
        *,
        local=True,
        archive_uncertainty=True,
        target_component=True
    ):
        self.prior = prior
        self.local = local
        self.kappa = kappa
        self.target_probability = target_probability if target_component else 0.0
        self.component_count = len(prior.names) + int(target_component)
        self.family = (
            prior.x[:, 4].astype(int) if local else np.zeros(len(prior.x), int)
        )
        self.blocks = {}
        self.positions = np.empty(len(prior.x), int)
        for family in np.unique(self.family):
            rows = np.flatnonzero(self.family == family)
            self.positions[rows] = np.arange(len(rows))
            means = prior.means[rows]
            probability = prior.weights * (1 - self.target_probability)
            groups = {}
            for j, archive in enumerate(prior.archive_covariances):
                key = id(archive) if archive_uncertainty else 0
                if key not in groups:
                    covariance = prior.enhanced_covariance[np.ix_(rows, rows)].copy()
                    if archive_uncertainty:
                        covariance += kappa * archive[np.ix_(rows, rows)]
                    groups[key] = ([], covariance)
                groups[key][0].append(j)
            geometry = list(groups.values())
            if target_component:
                means = np.column_stack((means, np.full(len(rows), 0.5)))
                probability = np.r_[probability, target_probability]
                geometry.append(
                    ([len(prior.names)], prior.physical_covariance[np.ix_(rows, rows)])
                )
            self.blocks[int(family)] = (
                rows,
                BlockPosterior(means, geometry, probability),
            )

    def probabilities(self):
        output = np.empty(len(self.family))
        for rows, block in self.blocks.values():
            output[rows] = block.probability(np.arange(len(rows)))
        return output

    def observe(self, index, risk):
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("The real oracle must disclose risk in [0,1]")
        self.blocks[int(self.family[index])][1].observe(self.positions[index], risk)

    def draws(self, indices, count, generator):
        output = np.empty((count, len(indices)))
        for family, (_, block) in self.blocks.items():
            columns = np.flatnonzero(self.family[indices] == family)
            if len(columns):
                output[:, columns] = block.draws(
                    self.positions[indices[columns]], count, generator
                )
        return output

    def losses(self, indices, reference):
        probability = self.probabilities()
        output = (1 - probability[indices]) * probability[reference]
        same = self.family[indices] == self.family[reference]
        block = self.blocks[int(self.family[reference])][1]
        output[same] = block.losses(
            self.positions[indices[same]], self.positions[reference]
        )
        return output

    def source_tokens(self, indices):
        output = np.empty((len(indices), self.component_count, 4))
        for family, (_, block) in self.blocks.items():
            rows = np.flatnonzero(self.family[indices] == family)
            mean, variance, beta = block.parameters(self.positions[indices[rows]])
            output[rows, :, 0] = mean
            output[rows, :, 1] = np.sqrt(variance * beta / (block.alpha - 1))
            output[rows, :, 2] = np.exp(block.log_weight)
            output[rows, :, 3] = 0
            if self.target_probability:
                output[rows, -1, 3] = 1
        return output

    def weights(self):
        return {
            str(family): np.exp(block.log_weight).tolist()
            for family, (_, block) in self.blocks.items()
        }


def model_set(prior, **options):
    return [
        LocalPosterior(prior, kappa, target_probability, **options)
        for kappa, target_probability in MODELS
    ]
