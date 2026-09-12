from __future__ import annotations

import numpy as np
import pytest

from mvr.metadrive.diva.factorization import fit_low_rank_vulnerability


def test_global_low_rank_factorization_recovers_rank_one_response() -> None:
    latent = np.asarray((-1.0, -0.3, 0.5, 1.2))
    basis = np.asarray((-0.4, 0.2, 0.7, -0.1, 0.3))
    mean = np.asarray((0.3, 0.4, 0.1, 0.5, 0.2))
    values = mean[None, :] + latent[:, None] * basis[None, :]
    result = fit_low_rank_vulnerability(values, np.ones_like(values, dtype=bool), ("a", "b", "c", "d"), tuple(str(i) for i in range(5)), 1)
    assert result.rank == 1
    for index in range(4):
        np.testing.assert_allclose(result.prediction_for_source(index), values[index], atol=1e-12)
    assert result.residual_variance < 1e-20


def test_factorization_excludes_ineligible_columns_and_rejects_excess_rank() -> None:
    values = np.arange(12, dtype=float).reshape(3, 4)
    eligible = np.ones_like(values, dtype=bool)
    eligible[1, 2] = False
    result = fit_low_rank_vulnerability(values, eligible, ("a", "b", "c"), ("0", "1", "2", "3"), 1)
    assert not result.common_eligible_mask[2]
    assert np.isnan(result.mean[2])
    with pytest.raises(ValueError):
        fit_low_rank_vulnerability(values, eligible, ("a", "b", "c"), ("0", "1", "2", "3"), 3)
