from __future__ import annotations

import numpy as np

from mvr.diva.basis_gp import BasisGPBank


def test_candidate_gp_interpolates_mean_and_global_basis_components() -> None:
    features = np.asarray(((0.0, 0.0), (0.3, 0.2), (0.7, 0.8), (1.0, 1.0)))
    bank = BasisGPBank(device="cpu", fit_steps=3)
    bank.fit(0, features, np.asarray((0.0, 0.2, 0.7, 1.0)), np.asarray(((0.0,), (0.1,), (0.4,), (0.8,))))
    mean, mean_var, basis, basis_var = bank.predict(0, np.asarray(((0.5, 0.5),)))
    assert mean.shape == mean_var.shape == (1,)
    assert basis.shape == basis_var.shape == (1, 1)
    assert mean_var[0] > 0.0 and basis_var[0, 0] > 0.0
