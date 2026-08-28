from mvr.scripts.sweep_stage1_actions import _summary


def _row(name: str, ttc: float, distance: float, failure: bool) -> dict[str, object]:
    return {
        "family": "cutin",
        "task_id": "cutin-g04-fast_small_gap",
        "case_index": 0,
        "residual_name": name,
        "scenario": {"candidate_id": "target", "episode_seed": 104},
        "outcome": {
            "is_valid_episode": True,
            "is_failure": failure,
            "valid_target_collision": False,
            "valid_critical_near_miss": failure,
            "min_ttc": ttc,
            "min_distance": distance,
        },
    }


def test_action_sweep_summary_keeps_fixed_x0_and_paired_effect() -> None:
    report = _summary((_row("base", 5.0, 12.0, False), _row("acceleration_press", 2.0, 4.0, True)))

    assert report["paired_initial_conditions_verified"]
    assert report["by_family"]["cutin"]["any_valid_critical"]
    assert report["paired_against_base"]["acceleration_press"]["mean_ttc_reduction_s"] == 3.0
