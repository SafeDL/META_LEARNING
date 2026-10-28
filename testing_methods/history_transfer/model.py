"""Joint collision and continuous RSS response kernel."""

import math

import torch
from torch import nn

from testing_methods.common.gp_classifier import ClassificationKernel

MARGIN_NOISE = 0.05


class JointResponseKernel(ClassificationKernel):
    def __init__(self, dimension):
        super().__init__(dimension, deep=True)
        with torch.no_grad():
            self.mean.fill_(-1.4)
        self.margin_mean = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.log_margin_scale = nn.Parameter(torch.tensor(math.log(0.3), dtype=torch.float64))
        self.raw_correlation = nn.Parameter(torch.tensor(math.atanh(-0.5), dtype=torch.float64))

    @property
    def correlation(self):
        return self.raw_correlation.tanh()

    def joint_mean(self, count):
        return torch.cat((self.mean.expand(count), self.margin_mean.expand(count)))

    def joint_covariance(self, x, other=None):
        base = self.covariance(x, other)
        scale = self.log_margin_scale.exp()
        cross = self.correlation * scale * base
        return torch.cat(
            (torch.cat((base, cross), dim=-1), torch.cat((cross, scale.square() * base), dim=-1)),
            dim=-2,
        )
