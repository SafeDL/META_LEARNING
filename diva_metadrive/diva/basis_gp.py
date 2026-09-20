"""Small exact Matérn-5/2 ARD GP bank for interpolating frozen DIVA components."""
from __future__ import annotations

from dataclasses import dataclass

import gpytorch
import numpy as np
import torch


class _MaternExactGP(gpytorch.models.ExactGP):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood) -> None:
        super().__init__(features, targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=features.shape[-1]))

    def forward(self, features: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(features),
                                                         self.covar_module(features))


@dataclass
class _ExactMaternGP:
    model: _MaternExactGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood

    @classmethod
    def fit(cls,
            features: np.ndarray,
            targets: np.ndarray,
            *,
            device: str,
            steps: int = 100) -> "_ExactMaternGP":
        target_device = torch.device(device)
        x = torch.as_tensor(features, dtype=torch.float64, device=target_device)
        y = torch.as_tensor(targets, dtype=torch.float64, device=target_device).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("exact GP requires at least two aligned training anchors")
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(target_device,
                                                                  dtype=torch.float64)
        model = _MaternExactGP(x, y, likelihood).to(target_device, dtype=torch.float64)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
        marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(steps):
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-6):
                loss = -marginal_log_likelihood(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()
        return cls(model, likelihood)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        device = self.model.train_inputs[0].device
        query = torch.as_tensor(features, dtype=torch.float64, device=device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
                1e-6):
            posterior = self.model(query)
        return posterior.mean.cpu().numpy(), posterior.variance.clamp_min(1e-10).cpu().numpy()


class BasisGPBank:
    """Fit candidate-specific GPs for one shared mean and globally aligned bases."""
    def __init__(self, device: str = "cuda", fit_steps: int = 100) -> None:
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        if fit_steps < 1:
            raise ValueError("GP fit_steps must be positive")
        self.fit_steps = int(fit_steps)
        self.models: dict[int, tuple[_ExactMaternGP, tuple[_ExactMaternGP, ...]]] = {}

    def fit(self, candidate_index: int, features: np.ndarray, mean_target: np.ndarray,
            basis_targets: np.ndarray) -> None:
        values = np.asarray(basis_targets, dtype=np.float64)
        if values.ndim != 2 or len(features) != len(mean_target) or len(features) != len(values):
            raise ValueError("GP component targets must align with features")
        mean_model = _ExactMaternGP.fit(features,
                                        mean_target,
                                        device=self.device,
                                        steps=self.fit_steps)
        basis_models = tuple(
            _ExactMaternGP.fit(
                features, values[:, index], device=self.device, steps=self.fit_steps)
            for index in range(values.shape[1]))
        self.models[int(candidate_index)] = (mean_model, basis_models)

    def predict(self, candidate_index: int,
                query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            mean_model, basis_models = self.models[int(candidate_index)]
        except KeyError as error:
            raise ValueError("candidate GP has not been fit") from error
        mean, mean_var = mean_model.predict(query)
        if not basis_models:
            return mean, mean_var, np.empty((len(mean), 0)), np.empty((len(mean), 0))
        predicted = [model.predict(query) for model in basis_models]
        return mean, mean_var, np.column_stack([row[0] for row in predicted
                                                ]), np.column_stack([row[1] for row in predicted])
