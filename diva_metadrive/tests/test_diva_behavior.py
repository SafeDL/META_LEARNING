from __future__ import annotations

import numpy as np

from diva_metadrive.diva.behavior import DivaCutInBehavior
from diva_metadrive.training.runner import InnerActionPhase


def test_diva_behavior_tracks_one_prescribed_speed_without_sut_feedback() -> None:
    behavior = DivaCutInBehavior()
    before_onset = InnerActionPhase(False, False, 0.0, 8.0, 15.0)
    at_target = InnerActionPhase(True, False, 0.5, 8.0, 15.0)
    below_target = InnerActionPhase(True, False, 0.5, 5.0, 15.0)
    completed = InnerActionPhase(True, True, 1.0, 8.0, 15.0)
    np.testing.assert_allclose(behavior(before_onset), (0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(behavior(at_target), (0.0, 0.0, 0.0, 0.0))
    assert behavior(below_target)[3] > 0.0
    np.testing.assert_allclose(behavior(completed), (0.0, 0.0, 0.0, 0.0))
    assert behavior(before_onset).dtype == np.float32
    assert behavior.prescribed_speed_mps == 8.0


def test_diva_behavior_does_not_change_a_prescribed_speed_for_maneuver_phase() -> None:
    behavior = DivaCutInBehavior()
    before = InnerActionPhase(False, False, 0.0, 8.0, 15.0)
    after = InnerActionPhase(True, True, 1.0, 8.0, 15.0)
    np.testing.assert_allclose(behavior(before), behavior(after))
