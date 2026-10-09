"""Predict scene responses from declared behavioral hypotheses only."""
import numpy as np
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression
import torch

from methods.history_guided_testing.io import read_json
from research.response_adaptive_testing.confirmation import CONFIRMATION

PARTICLE_LOG2 = 8
PREDICTION_BATCH = 65536


def behavior_grid(log2=PARTICLE_LOG2):
    grid = qmc.Sobol(3, scramble=True, seed=20261017).random_base2(log2)
    return np.concatenate((np.column_stack((grid, np.zeros(len(grid)))),
                           np.column_stack(
                               (grid, np.ones(len(grid)))))).astype(np.float32)


def historical_risk_calibrators(names):
    values = {family: [[], []] for family in (0, 1)}
    for name in names:
        for replicate in range(2):
            with np.load(CONFIRMATION / name / f"pool_{replicate}" /
                         "responses.npz") as bank:
                for family in (0, 1):
                    chosen = bank["x"][:, 4] == family
                    values[family][0].append(bank["risk"][chosen])
                    values[family][1].append(bank["collision"][chosen])
    result = {}
    for family, (risk, collision) in values.items():
        model = LogisticRegression(C=100).fit(
            np.concatenate(risk)[:, None], np.concatenate(collision))
        result[family] = {
            "slope": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0])
        }
    return result


def response_tables(x, behaviors, models, return_logits=False):
    device = "cuda"
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    behaviors = torch.as_tensor(behaviors, dtype=torch.float32, device=device)
    risks = x.new_empty((len(behaviors), len(x)))
    collisions = torch.empty_like(risks)
    with torch.inference_mode():
        for family in (0, 1):
            indices = torch.nonzero(x[:, 4] == family, as_tuple=True)[0]
            risk_values = x.new_empty((len(behaviors) * len(indices), ))
            collision_values = torch.empty_like(risk_values)
            for start in range(0, len(risk_values), PREDICTION_BATCH):
                flat = torch.arange(start,
                                    min(start + PREDICTION_BATCH,
                                        len(risk_values)),
                                    device=device)
                inputs = torch.cat((x[indices[flat % len(indices)], :4],
                                    behaviors[flat // len(indices)]), 1)
                r, logits = models[family](inputs)
                risk_values[flat] = r
                collision_values[flat] = (logits if return_logits else
                                          logits.sigmoid())
            risks[:, indices] = risk_values.reshape(len(behaviors), -1)
            collisions[:,
                       indices] = collision_values.reshape(len(behaviors), -1)
    return risks, collisions
