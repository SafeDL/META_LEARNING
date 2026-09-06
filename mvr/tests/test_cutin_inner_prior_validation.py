from types import SimpleNamespace

from mvr.scripts.evaluate_cutin_inner_prior_validation import (
    random_cases,
    select_render_case_ids,
    summarize,
)


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        functional_scenario="cutin",
        logical_domain_bounds={
            "cutin_gap_at_start_m": (-0.5, -0.2),
            "sut_initial_speed_mps": (0.2, 0.5),
            "relative_speed_mps": (0.1, 0.4),
            "cutin_start_progress": (-0.4, -0.1),
            "cutin_start_time_s": (0.2, 0.5),
        },
    )


def _record(case_index: int, policy: str, *, failure: bool, ttc: float) -> dict[str, object]:
    return {
        "case_index": case_index,
        "policy": policy,
        "valid": True,
        "failure": failure,
        "event": failure,
        "target_collision": failure,
        "critical_near_miss": False,
        "min_ttc": ttc,
        "min_distance": 8.0,
        "raw_near_miss_candidate_steps": 2,
        "valid_event_capture_steps": int(failure),
    }


def test_random_cases_are_reproducible_and_stay_within_validation_bounds() -> None:
    task = _task()

    first = random_cases(task, 4, 17)
    second = random_cases(task, 4, 17)

    assert first == second
    for action in first:
        assert action.candidate_index in {0, 1}
        for value, bounds in zip(action.continuous, task.logical_domain_bounds.values()):
            assert bounds[0] <= value <= bounds[1]


def test_summary_and_render_selection_use_paired_failure_contracts() -> None:
    rows = [
        _record(0, "sac", failure=False, ttc=2.0),
        _record(0, "random", failure=True, ttc=1.0),
        _record(1, "sac", failure=False, ttc=4.0),
        _record(1, "random", failure=False, ttc=3.0),
    ]

    report = summarize([row for row in rows if row["policy"] == "random"])

    assert report["failure_count"] == 1
    assert report["failure_rate"] == 0.5
    assert report["raw_near_miss_candidate_steps"] == 4
    assert report["valid_event_capture_steps"] == 1
    assert select_render_case_ids(rows) == [0, 1]
