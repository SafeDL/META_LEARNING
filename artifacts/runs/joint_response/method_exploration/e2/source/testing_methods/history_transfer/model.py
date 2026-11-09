"""Joint collision and continuous RSS response kernel."""

import math

import torch
from torch import nn

from testing_methods.common.gp_classifier import ClassificationKernel

MARGIN_NOISE = 0.05


class JointResponseKernel(ClassificationKernel):
    def __init__(self, dimension, *, deep=True):
        super().__init__(dimension, deep=deep)
        with torch.no_grad():
            self.mean.fill_(-1.4)
        self.margin_mean = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.log_margin_scale = nn.Parameter(torch.tensor(math.log(0.3), dtype=torch.float64))
        self.raw_correlation = nn.Parameter(torch.tensor(math.atanh(-0.5), dtype=torch.float64))

    @property
    def correlation(self):
        return self.raw_correlation.tanh()

    def output_covariance(self):
        scale = self.log_margin_scale.exp()
        one = torch.ones_like(scale)
        cross = self.correlation * scale
        return torch.stack((torch.stack((one, cross)), torch.stack((cross, scale.square()))))

    def point_mean(self, count):
        return torch.stack((self.mean, self.margin_mean)).expand(count, 2)

    def point_covariance(self, count):
        return (self.log_scale.exp().square() * self.output_covariance()).expand(count, 2, 2)

    def source_covariance(self, covariance):
        return covariance

    def adaptation_groups(self):
        return [[p for p in self.parameters() if p.requires_grad]]

    def kernel(self, x, other=None):
        """Interleaved (collision, RSS) coordinates, without a query-query matrix."""
        z = self.features(x)
        return self.kernel_from_features(z, None if other is None else self.features(other))

    def kernel_from_features(self, z, other=None):
        """Use current kernel scalars with features of the fixed shared network."""
        other = z if other is None else other
        distance = torch.cdist(z, other) / self.log_lengthscale.exp().clamp_min(0.01)
        r = math.sqrt(5) * distance
        base = self.log_scale.exp().square() * (1 + r + r.square() / 3) * (-r).exp()
        return torch.einsum("ij,ab->iajb", base, self.output_covariance()).reshape(
            2 * len(z), 2 * len(other)
        )
