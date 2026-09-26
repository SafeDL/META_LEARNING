"""Profiled FVDM source-schema contract without launching a simulator."""

import numpy as np

from methods.core_mine import fvdm_source_robustness as study


def test_fvdm_source_uses_matching_safety_fields(monkeypatch) -> None:
    observed = []

    def fake_episode(profile, scenario, seed):
        observed.append((profile.name, scenario.mode, seed))
        return {"ego_collision": False, "background_collision": False,
                "near_miss": False, "min_ttc": 2.0,
                "min_clearance": 4.0, "completed": True}

    monkeypatch.setattr(study, "_episode", fake_episode)
    name, values = study._source_job((
        7, "fvdm_ref", np.asarray([[20.0, -3.0]]),
        np.asarray(["fast_intrusion"]), np.asarray([[.2, .3]])))
    assert name == "fvdm_ref"
    assert observed == [("SM-Strong-FVDM", "fast_intrusion", 7)]
    assert values["min_distance"].tolist() == [4.0]
    assert values["completed"].tolist() == [True]
