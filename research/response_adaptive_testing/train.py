"""Learn joint transfer parameters from held-group historical risk contexts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.history_guided_testing.config import GROUPS, ROOT as BASELINE
from methods.history_guided_testing.io import write_json

from .config import OUTPUT
from .history_validation import predictions
from .model import joint_prior, risk_coordinate

STEPS = 1200
CONTEXT_COUNTS = (10, 30, 60, 100)
QUERY_COUNT = 192


class TransferParameters(nn.Module):

    def __init__(self, transform, positive_loading=False):
        super().__init__()
        self.transform = transform
        self.positive_loading = positive_loading
        initial = ([0.1, 0.25, 1, 0.0025, 0.05, 0.0025]
                   if transform == "raw" else [0.1, 1, 0.25, 1, 0.5, 1])
        self.log_parameters = nn.Parameter(
            torch.tensor(np.log(initial), dtype=torch.float64))
        loading = torch.zeros((2, 6), dtype=torch.float64)
        loading[:, -1] = 1
        self.loading = nn.Parameter(loading)
        self.offset = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def options(self):
        values = self.log_parameters.exp()
        return {
            "positive_loading":
            self.positive_loading,
            "transform":
            self.transform,
            "collision_transform":
            "probit",
            "residual_mode":
            "learned_sensitivity",
            "lookahead":
            False,
            **dict(
                zip(("source_scale", "risk_scale", "collision_scale", "noise", "length", "sensitivity_regularization"), values)), "loading":
            self.loading,
            "collision_offset":
            self.offset
        }

    def serializable(self):
        return {
            key: (value.detach().cpu().tolist()
                  if torch.is_tensor(value) else value)
            for key, value in self.options().items()
        }


def episode(model, task, rng):
    x, risks, collisions, truth_r, truth_c = task
    context_count = int(rng.choice(CONTEXT_COUNTS))
    available = np.arange(len(x))
    # Histories emulate both risk-directed and broad sampling, using no target label for selection.
    if rng.integers(2):
        proposal = np.argsort(-collisions.mean(1))[:600]
        context = rng.choice(proposal, context_count, replace=False)
    else:
        context = rng.choice(available, context_count, replace=False)
    remaining = np.setdiff1d(available, context)
    positive = remaining[truth_c[remaining]]
    negative = remaining[~truth_c[remaining]]
    count = min(QUERY_COUNT // 2, len(positive))
    query = np.concatenate((rng.choice(positive, count, replace=False),
                            rng.choice(negative,
                                       QUERY_COUNT - count,
                                       replace=False)))
    indices = np.concatenate((context, query))
    options = model.options()
    mr, mc, rr, cr, cc, _ = joint_prior(x[indices], risks[indices],
                                        collisions[indices], options)
    kernel = rr[:context_count, :context_count] + options["noise"] * torch.eye(
        context_count, dtype=rr.dtype, device=rr.device)
    cross = cr[context_count:, :context_count]
    observed = mr.new_tensor(risk_coordinate(truth_r[context],
                                             model.transform))
    posterior_mean = mc[context_count:] + cross @ torch.linalg.solve(
        kernel, observed - mr[:context_count])
    posterior_variance = (
        cc[context_count:] -
        (cross * torch.linalg.solve(kernel, cross.T).T).sum(1)).clamp(min=0)
    q = torch.special.ndtr(posterior_mean /
                           (1 + posterior_variance).sqrt()).clamp(
                               1e-8, 1 - 1e-8)
    score = torch.logit(q)
    ranking = F.softplus(score[count:][None, :] - score[:count, None]).mean()
    prevalence = float(truth_c[remaining].mean())
    calibration = (-prevalence * q[:count].log().mean() - (1 - prevalence) *
                   (1 - q[count:]).log().mean())
    return ranking + 0.1 * calibration


def main():
    torch.set_num_threads(1)
    bank = np.load(BASELINE / "history/responses.npz")
    tasks = []
    for group, targets in GROUPS.items():
        _, risks, collisions = predictions(bank, group)
        for target in targets:
            tasks.append((bank["x"], risks, collisions, bank[target],
                          bank[target + "_collision"]))
    for transform in ("raw", "logit"):
        torch.manual_seed(11)
        model = TransferParameters(transform, positive_loading=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        rng = np.random.default_rng(20261005)
        curve = []
        for step in range(STEPS):
            loss = episode(model, tasks[int(rng.integers(len(tasks)))], rng)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            curve.append(float(loss.detach()))
            if (step + 1) % 100 == 0:
                options = model.serializable()
                write_json(
                    OUTPUT / "models" / "monotone_transfer" /
                    f"{transform}_{step + 1}.json", {
                        "options":
                        options,
                        "curve":
                        curve,
                        "steps":
                        step + 1,
                        "target_data_used":
                        False,
                        "target_groups_excluded_from_predictors":
                        True,
                        "context_sampling":
                        "independent task, size and sampling policy draws",
                        "calibration":
                        "population-prevalence corrected case-control loss",
                        "scope":
                        "historical meta-training; not independent confirmation"
                    })
                print("META",
                      transform,
                      step + 1,
                      round(float(np.mean(curve[-100:])), 4),
                      round(options["noise"], 4),
                      round(options["length"], 4),
                      flush=True)


if __name__ == "__main__":
    main()
