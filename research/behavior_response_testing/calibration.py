"""Estimate conditional risk model discrepancy from historical risk residuals."""
import math

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.kernel import matern52
from research.response_adaptive_testing.confirmation import CONFIRMATION

from .model import BehaviorResponseModel, behavior_coordinates, response_inputs
from .session import rmse_discrepancy
from .train import OUTPUT, SEEDS

CONTEXT_SIZE = 128
STEPS = 400


def risk_error_nll(distance, residual, log_parameters):
    gp_variance, noise_variance, length = log_parameters.exp()
    covariance = gp_variance * matern52(distance, length)
    covariance = covariance + noise_variance * torch.eye(
        distance.shape[-1], dtype=distance.dtype, device=distance.device)
    factor = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(residual[..., None], factor).squeeze(-1)
    quadratic = (residual * solved).sum(-1)
    log_determinant = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return (0.5 * (quadratic + log_determinant +
                   residual.shape[-1] * math.log(2 * math.pi)) /
            residual.shape[-1]).mean()


def main():
    torch.set_num_threads(1)
    training = read_json(OUTPUT / "protocol.json")
    profiles = {
        p["name"]: p
        for p in read_json(CONFIRMATION / "protocol.json")["profiles"]
    }
    write_json(
        OUTPUT / "calibration" / "protocol.json", {
            "role": "Historical known-profile risk discrepancy estimation",
            "profiles": training["validation_profiles"],
            "seed": 20261018,
            "contexts_per_family": 16,
            "context_size": CONTEXT_SIZE,
            "steps": STEPS,
            "optimizer":
            "Adam lr=0.03 on log GP variance, noise variance and length",
            "feedback":
            "Historical continuous risk; no collision label in objective",
            "target_access": "Known historical parameters used only offline"
        })
    for seed in SEEDS:
        destination = OUTPUT / "calibration" / f"discrepancy_{seed}.json"
        if destination.exists():
            continue
        states = torch.load(OUTPUT / "models" / f"predictor_{seed}.pt",
                            map_location="cuda",
                            weights_only=True)
        validation = read_json(
            OUTPUT / "models" /
            f"predictor_{seed}.json")["known_profile_validation"]
        initial = rmse_discrepancy(
            [validation[str(f)]["risk_rmse"]**2 for f in (0, 1)])
        fitted = {key: [] for key in initial}
        summaries = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            distances, errors = [], []
            rng = np.random.default_rng(20261018)
            with torch.inference_mode():
                for name in training["validation_profiles"]:
                    behavior = behavior_coordinates(profiles[name])
                    for replicate in range(2):
                        with np.load(CONFIRMATION / name /
                                     f"pool_{replicate}" /
                                     "responses.npz") as bank:
                            chosen = rng.choice(
                                np.flatnonzero(bank["x"][:, 4] == family),
                                CONTEXT_SIZE,
                                replace=False)
                            inputs = torch.as_tensor(response_inputs(
                                bank["x"][chosen], behavior),
                                                     device="cuda")
                            prediction = model(inputs)[0]
                            risk = torch.as_tensor(bank["risk"][chosen],
                                                   dtype=torch.float64,
                                                   device="cuda")
                            errors.append((risk - prediction).cpu().numpy())
                            x = inputs[:, :4].double()
                            distances.append(torch.cdist(x, x).cpu().numpy())
            distance = torch.as_tensor(np.array(distances), device="cuda")
            residual = torch.as_tensor(np.array(errors), device="cuda")
            parameters = torch.tensor(
                [initial[key][family] for key in initial],
                dtype=torch.float64,
                device="cuda").log()
            parameters.requires_grad_()
            optimizer = torch.optim.Adam([parameters], lr=0.03)
            before = float(
                risk_error_nll(distance, residual, parameters).detach())
            for step in range(1, STEPS + 1):
                optimizer.zero_grad()
                loss = risk_error_nll(distance, residual, parameters)
                loss.backward()
                optimizer.step()
            after = float(
                risk_error_nll(distance, residual, parameters).detach())
            for key, value in zip(fitted, parameters.detach().exp().tolist()):
                fitted[key].append(value)
            summaries.append({
                "family": family,
                "initial_nll": before,
                "fitted_nll": after
            })
        write_json(destination, {"parameters": fitted, "fit": summaries})
        print("RISK DISCREPANCY", seed, fitted, summaries, flush=True)


if __name__ == "__main__":
    main()
