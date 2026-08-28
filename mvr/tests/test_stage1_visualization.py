from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mvr.scripts.visualize_stage1 import _compact_outcome, _dual_view_frame, select_representative


def test_representative_selection_prefers_valid_critical_episode() -> None:
    safe = SimpleNamespace(outcome={
        "is_failure": False,
        "target_collision": False,
        "valid_critical_near_miss": False,
        "min_ttc": 0.1,
        "max_closing_speed": 30.0,
        "min_distance": 0.1,
    })
    critical = SimpleNamespace(outcome={
        "is_failure": True,
        "target_collision": False,
        "valid_critical_near_miss": True,
        "min_ttc": 1.0,
        "max_closing_speed": 4.0,
        "min_distance": 2.0,
    })
    assert select_representative((safe, critical)) is critical


def test_representative_selection_does_not_promote_invalid_collision() -> None:
    invalid_collision = SimpleNamespace(outcome={
        "is_valid_episode": False,
        "is_failure": False,
        "target_collision": True,
        "valid_critical_near_miss": False,
        "min_ttc": 0.1,
        "max_closing_speed": 30.0,
        "min_distance": 0.1,
    })
    valid_safe = SimpleNamespace(outcome={
        "is_valid_episode": True,
        "is_failure": False,
        "target_collision": False,
        "valid_critical_near_miss": False,
        "min_ttc": 2.0,
        "max_closing_speed": 4.0,
        "min_distance": 2.0,
    })
    assert select_representative((invalid_collision, valid_safe)) is valid_safe


def test_dual_view_frame_preserves_two_views() -> None:
    frame = _dual_view_frame(
        np.full((20, 30, 3), 20, dtype=np.uint8),
        np.full((20, 40, 3), 40, dtype=np.uint8),
        "trained_inner",
        "main_conflict",
        3,
        "lane following",
    )
    assert frame.shape == (20, 76, 3)


def test_compact_outcome_excludes_detailed_telemetry() -> None:
    compact = _compact_outcome({
        "is_valid_episode": True,
        "min_ttc": 1.5,
        "traffic_telemetry": {"steps": [1, 2, 3]},
    })
    assert compact == {"is_valid_episode": True, "min_ttc": 1.5}
