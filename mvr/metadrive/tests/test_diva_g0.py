from __future__ import annotations

import numpy as np

from mvr.metadrive.diva.source_bank import SourceBank
from mvr.metadrive.diva.types import DivaCutInDesign
from mvr.metadrive.scripts.analyze_diva_prior import analyze_bank


CONFIG = {
    "gates": {
        "g0": {
            "min_formal_valid_rate": 0.95,
            "min_common_eligible_per_candidate": 24,
            "min_pooled_response_std": 0.05,
            "min_source_response_range": 0.10,
            "min_sources_meeting_response_range": 3,
            "min_anchor_disagreement_p75": 0.03,
            "min_rank2_explained_variance": 0.80,
        }
    }
}


def _bank(responses: np.ndarray, formal: np.ndarray | None = None) -> SourceBank:
    designs = tuple(
        DivaCutInDesign(candidate, (index / 40.0,) * 5)
        for candidate in (0, 1)
        for index in range(24)
    )
    return SourceBank(
        ("a", "b", "c", "d"),
        designs,
        np.zeros_like(responses) if formal is None else formal,
        responses,
        np.ones_like(responses, dtype=bool),
        (),
    )


def test_g0_rejects_degenerate_and_one_source_only_responses() -> None:
    constant = analyze_bank(_bank(np.full((4, 48), 0.2)), CONFIG, "v2")
    one_source = np.zeros((4, 48))
    one_source[0] = np.linspace(0.0, 1.0, 48)
    sparse = analyze_bank(_bank(one_source), CONFIG, "v2")
    assert not constant["g0_structural_viability"]["pass"]
    assert not sparse["g0_structural_viability"]["pass"]


def test_g0_accepts_informative_continuous_response_without_formal_events() -> None:
    basis = np.linspace(-0.25, 0.25, 48)
    latent = np.asarray((-1.0, -0.3, 0.5, 1.2))
    responses = 0.5 + latent[:, None] * basis[None, :]
    report = analyze_bank(_bank(responses), CONFIG, "v2")
    assert report["summary"]["formal_event_rate"] == 0.0
    assert report["g0_structural_viability"]["pass"]
    assert report["rank2_cumulative_explained_variance"] == 1.0
