"""A shared latent process and two output-private processes for collision/RSS."""

import math

import torch
from torch import nn

from .model import JointResponseKernel


class SharedPrivateKernel(JointResponseKernel):
    """f=sf*(sqrt(q)*u+sqrt(1-q)*v), r=sr*(a*u+sqrt(1-a²)*w)."""

    source_path = "testing_methods/history_transfer/shared_private.py"

    def __init__(self, dimension, *, deep=True):
        super().__init__(dimension, deep=deep)
        self.raw_shared_fraction = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_collision_lengthscale = nn.Parameter(
            torch.tensor(math.log(0.3), dtype=torch.float64)
        )
        self.log_rss_lengthscale = nn.Parameter(torch.tensor(math.log(0.3), dtype=torch.float64))

    @property
    def shared_fraction(self):
        return self.raw_shared_fraction.sigmoid()

    @property
    def shared_loading(self):
        return self.raw_correlation.tanh()

    @property
    def correlation(self):
        return self.shared_fraction.sqrt() * self.shared_loading

    def output_covariance(self):
        # RSS has an absolute scale so that zero loading also separates its
        # scale hyperparameter from the collision scale.
        relative_scale = (self.log_margin_scale - self.log_scale).exp()
        one = torch.ones_like(relative_scale)
        cross = self.correlation * relative_scale
        return torch.stack(
            (torch.stack((one, cross)), torch.stack((cross, relative_scale.square())))
        )

    def kernel_from_features(self, z, other=None):
        other = z if other is None else other
        distance = torch.cdist(z, other)

        def matern(log_length):
            scaled = math.sqrt(5) * distance / log_length.exp().clamp_min(0.01)
            return (1 + scaled + scaled.square() / 3) * (-scaled).exp()

        shared = matern(self.log_lengthscale)
        collision = matern(self.log_collision_lengthscale)
        rss = matern(self.log_rss_lengthscale)
        fraction, loading = self.shared_fraction, self.shared_loading
        sf, sr = self.log_scale.exp(), self.log_margin_scale.exp()
        ff = sf.square() * (fraction * shared + (1 - fraction) * collision)
        rr = sr.square() * (loading.square() * shared + (1 - loading.square()) * rss)
        cross = sf * sr * fraction.sqrt() * loading * shared
        return torch.stack(
            (torch.stack((ff, cross), dim=-1), torch.stack((cross, rr), dim=-1)), dim=1
        ).reshape(2 * len(z), 2 * len(other))

    def source_covariance(self, covariance):
        """Contract source cross-output covariance without changing its marginals.

        This is g*C+(1-g)*blockdiag(C_ff,C_rr), g=abs(a), hence PSD.
        Applying it to every source also removes RSS influence when a=0.
        """
        rows = torch.arange(covariance.shape[-2], device=covariance.device) % 2
        columns = torch.arange(covariance.shape[-1], device=covariance.device) % 2
        multiplier = torch.where(rows[:, None] == columns[None, :], 1.0, self.shared_loading.abs())
        return covariance * multiplier

    def adaptation_groups(self):
        groups = [[], []]
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                rss_only = name in ("margin_mean", "log_margin_scale")
                groups[int(rss_only)].append(parameter)
        return [group for group in groups if group]
