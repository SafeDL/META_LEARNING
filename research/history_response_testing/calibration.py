"""Historical calibration; target outcomes are never used for fitting."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.history import load_history, split_indices


def calibrators():
    bank = np.load(BASELINE / "history/responses.npz")
    history = load_history()
    train, _ = split_indices(bank["x"])
    result = {}
    for family in (0, 1):
        indices = train[bank["x"][train, 4] == family]
        risk = np.concatenate([bank[name][indices] for name in history])[:,
                                                                         None]
        collision = np.concatenate(
            [bank[name + "_collision"][indices] for name in history])
        model = LogisticRegression(C=100).fit(risk, collision)
        result[family] = {
            "slope": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0])
        }
    return result
