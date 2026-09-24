from __future__ import annotations

import numpy as np
import pytest

from metadrive_benchmark.mining.types import CutInDesign


def test_mining_design_round_trip_and_identity_free_feature() -> None:
    design = CutInDesign(1, (-0.2, -0.1, 0.0, 0.1, 0.2))
    assert design.scenario_action().candidate_index == 1
    np.testing.assert_allclose(design.feature_vector(), (0.4, 0.45, 0.5, 0.55, 0.6))
    assert len(design.design_id) == 64
    assert "sut" not in design.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"candidate_index": 2, "logical_continuous": (0.0,) * 5},
        {"candidate_index": 0, "logical_continuous": (0.0, 0.0, 0.0, 0.0, 1.1)},
    ),
)
def test_mining_design_rejects_out_of_contract_values(kwargs) -> None:
    with pytest.raises(ValueError):
        CutInDesign(**kwargs)
