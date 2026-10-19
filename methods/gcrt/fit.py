"""Two historical empirical-Bayes parameters, selected without target feedback."""

import numpy as np
import torch

from methods.baselines.hgt.config import group_of
from methods.baselines.hgt.history import split_indices, subset
from methods.baselines.hgt.io import write_json
from methods.common.features import assemble
from methods.common.prior import visible_history

from .config import ROOT, SEED


def cases(workspace, split):
    history = workspace.history
    x = next(iter(history.values()))["x"]
    train, validation = split_indices(x)
    pool = train if split == "train" else validation
    count = 160 if split == "train" else 40
    selected = np.random.default_rng(SEED).permutation(pool)[:count]
    output = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in history:
        for condition in workspace.archives:
            sources = subset(history, train, excluded_group=group_of(name))
            sources = visible_history(sources, condition, SEED)
            _, prior = assemble(x[selected], sources, SEED, workspace=workspace)
            tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
            output.append(
                {
                    "group": group_of(name),
                    "base": tensor(prior.enhanced_covariance),
                    "archive": tensor(np.stack(prior.archive_covariances)),
                    "physical": tensor(prior.physical_covariance),
                    "mean": tensor(np.column_stack((prior.means, np.full(count, 0.5)))),
                    "weights": tensor(prior.weights),
                    "truth": tensor(history[name]["risk"][selected]),
                }
            )
    return output


def objective(parameters, tasks):
    kappa, target = parameters[0].exp(), parameters[1].sigmoid()
    grouped = {}
    for task in tasks:
        n = len(task["truth"])
        cov = torch.cat(
            (task["base"][None] + kappa * task["archive"], task["physical"][None]), 0
        )
        cov = cov + 0.00250001 * torch.eye(n, dtype=cov.dtype, device=cov.device)
        factor = torch.linalg.cholesky(cov)
        residual = (task["truth"][:, None] - task["mean"]).T[..., None]
        solved = torch.linalg.solve_triangular(factor, residual, upper=False)
        quadratic = solved.square().sum((-2, -1))
        logdet = factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        alpha, beta = parameters.new_tensor(2.5), parameters.new_tensor(1.5)
        density = (
            torch.lgamma(alpha + n / 2)
            - torch.lgamma(alpha)
            + alpha * beta.log()
            - (alpha + n / 2) * (beta + quadratic / 2).log()
            - n / 2 * np.log(2 * np.pi)
            - logdet
        )
        weights = torch.cat((task["weights"] * (1 - target), target[None]))
        loss = -torch.logsumexp(weights.log() + density, 0) / n
        grouped.setdefault(task["group"], []).append(loss)
    return torch.stack([torch.stack(rows).mean() for rows in grouped.values()]).mean()


def fit(workspace):
    training, validation = cases(workspace, "train"), cases(workspace, "validation")
    parameter = torch.nn.Parameter(
        training[0]["truth"].new_tensor([0.0, np.log(1 / 3)])
    )
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    best, selected, curve = None, None, []
    for step in range(101):
        with torch.no_grad():
            score = float(objective(parameter, validation).cpu())
        row = {
            "step": step,
            "validation_nll": score,
            "kappa": float(parameter[0].exp().detach().cpu()),
            "target_probability": float(parameter[1].sigmoid().detach().cpu()),
        }
        curve.append(row)
        if best is None or score < best:
            best = score
            selected = {k: row[k] for k in ("kappa", "target_probability")}
        if step == 100:
            break
        optimizer.zero_grad()
        loss = objective(parameter, training)
        loss.backward()
        optimizer.step()
    write_json(
        ROOT / workspace.root.name / "prior.json",
        {
            "selected": selected,
            "curve": curve,
            "physical_calls": 0,
            "objective": "mechanism-balanced per-observation joint Student mixture NLL",
        },
    )
