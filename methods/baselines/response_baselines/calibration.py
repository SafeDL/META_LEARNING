"""Historical calibration; target outcomes are never used for fitting."""
from benchmark.data import HISTORY

import numpy as np
from sklearn.linear_model import LogisticRegression

from methods.baselines.hgt.config import ROOT as BASELINE
from methods.baselines.hgt.history import load_history, split_indices


def calibrators():
    bank = np.load(HISTORY / "responses.npz")
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
