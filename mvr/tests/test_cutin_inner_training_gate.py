from __future__ import annotations

from mvr.scripts.evaluate_cutin_inner_training_gate import (
    paired_cases,
    summarize_records,
)
from mvr.training.pipeline import load_config


def _record(domain: str, *, valid: bool = True, event: bool = False) -> dict[str, object]:
    return {
        "logical_domain_id": domain,
        "valid": valid,
        "event": event,
        "cutin_path_or_event": event,
    }


def test_gate_rejects_missing_event_in_one_logical_domain() -> None:
    rows = [
        *[_record("close", event=True) for _ in range(3)],
        *[_record("balanced", event=True) for _ in range(3)],
        *[_record("late") for _ in range(3)],
    ]

    report = summarize_records(rows)

    assert report["summary"]["valid_event_rate"] == 2.0 / 3.0
    assert report["domains"]["late"]["event_rate"] == 0.0
    assert report["domains"]["late"]["passed"] is False
    assert report["passed_domain_count"] == 2
    assert report["total_domain_count"] == 3
    assert report["status"] == "partially_passed"
    assert report["passed"] is False


def test_gate_passes_when_every_logical_domain_has_an_event() -> None:
    rows = [
        *[_record("close", event=True) for _ in range(3)],
        _record("balanced", event=True), _record("balanced"), _record("balanced"),
        _record("late", event=True), _record("late"), _record("late"),
    ]

    report = summarize_records(rows)

    assert report["passed_domain_count"] == 3
    assert report["status"] == "passed"
    assert report["passed"] is True


def test_gate_fails_domain_below_minimum_valid_rate() -> None:
    rows = [
        *[_record("close", event=True) for _ in range(3)],
        _record("balanced", event=True), _record("balanced", valid=False),
        _record("balanced", valid=False),
    ]

    report = summarize_records(rows)

    assert report["domains"]["balanced"]["valid_rate"] == 1.0 / 3.0
    assert report["domains"]["balanced"]["passed"] is False
    assert report["status"] == "partially_passed"


def test_paired_cases_cover_three_fixed_cases_per_training_domain() -> None:
    config, taskbook, _ = load_config("mvr/configs/cutin_inner.yaml")

    cases = paired_cases(config, taskbook)

    assert len(cases) == 9
    assert {case["logical_domain_id"] for case in cases} == {
        "close_closing_early",
        "balanced_interaction",
        "late_tight_cutin",
    }
    for domain in {case["logical_domain_id"] for case in cases}:
        rows = [case for case in cases if case["logical_domain_id"] == domain]
        assert [row["source"] for row in rows] == [
            "centre", "domain_sample_1", "domain_sample_2",
        ]
        assert len({row["episode_seed"] for row in rows}) == 3
