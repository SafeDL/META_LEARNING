"""Exclude whole mechanism groups from network gradients, priors and calibration."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from methods.history_guided_testing.config import GROUPS, ROOT as BASELINE, group_of
from methods.history_guided_testing.history import split_indices
from methods.history_guided_testing.io import write_json

from .config import BUDGET, OUTPUT
from .history_model import fit_history_model
from .kernel import covariance
from .session import RiskTestingSession


def main():
    torch.set_num_threads(1)
    bank = np.load(BASELINE / "history/responses.npz")
    train, validation = split_indices(bank["x"])
    records = []
    curves = {}
    for excluded in GROUPS:
        names = [
            n for g, members in GROUPS.items() if g != excluded
            for n in members
        ]
        assert all(group_of(n) != excluded for n in names)
        weights = np.asarray([1 / 3 / len(GROUPS[group_of(n)]) for n in names])
        predicted_risk = np.zeros((len(validation), len(names)))
        predicted_collision = np.zeros_like(predicted_risk)
        calibration = {}
        torch.manual_seed(20261005)
        for family in (0, 1):
            train_indices = train[bank["x"][train, 4] == family]
            validation_indices = validation[bank["x"][validation, 4] == family]
            positions = np.flatnonzero(bank["x"][validation, 4] == family)
            model, curve = fit_history_model(
                bank,
                names,
                train_indices,
                validation_indices,
                np.random.default_rng(20261005),
            )
            with torch.no_grad():
                risk, logits = model(
                    torch.as_tensor(
                        bank["x"][validation_indices, :4],
                        device="cuda",
                        dtype=torch.float32,
                    ))
                output = torch.cat((risk, logits.sigmoid()),
                                   dim=1).cpu().numpy()
            predicted_risk[positions] = output[:, :len(names)]
            predicted_collision[positions] = output[:, len(names):]
            flat_risk = np.concatenate([bank[n][train_indices]
                                        for n in names])[:, None]
            flat_collision = np.concatenate(
                [bank[n + "_collision"][train_indices] for n in names])
            fitted = LogisticRegression(C=100).fit(flat_risk, flat_collision)
            calibration[family] = {
                "slope": float(fitted.coef_[0, 0]),
                "intercept": float(fitted.intercept_[0])
            }
            curves[f"{excluded}:{family}"] = curve
        x = bank["x"][validation]
        kernel = covariance(x, predicted_risk)
        for target in GROUPS[excluded]:
            session = RiskTestingSession(x, kernel, predicted_risk,
                                         predicted_collision, weights,
                                         calibration)
            selected = []
            while (index := session.next_index()) is not None:
                session.observe(float(bank[target][validation[index]]))
                selected.append(index)
            collision = bank[target + "_collision"][validation]
            curve = np.cumsum(collision[selected])
            ranking = np.argsort(-(predicted_collision @ weights),
                                 kind="stable")[:BUDGET]
            ranking_curve = np.cumsum(collision[ranking])
            records.append({
                "excluded_group": excluded,
                "target": target,
                "available_sources": names,
                "target_excluded_from_gradients": True,
                "target_excluded_from_checkpoint_selection": True,
                "target_excluded_from_calibration": True,
                "pool_collision_count": int(collision.sum()),
                "candidate_F50": int(curve[49]),
                "candidate_F100": int(curve[99]),
                "candidate_F200": int(curve[-1]),
                "candidate_auc": float(curve.mean()),
                "ranking_auc": float(ranking_curve.mean()),
                "ranking_F200": int(ranking_curve[-1]),
                "selected_indices": selected
            })
            print("Excluded-group",
                  excluded,
                  target,
                  records[-1]["candidate_F200"],
                  "/",
                  int(collision.sum()),
                  flush=True)
    write_json(
        OUTPUT / "group_validation.json", {
            "records":
            records,
            "training_curves":
            curves,
            "independent_group_count":
            4,
            "validation_coordinates":
            410,
            "budget":
            BUDGET,
            "scope":
            "diagnostic with fixed developed hyperparameters, not a new blind confirmation or six independent tasks"
        })


if __name__ == "__main__":
    main()
