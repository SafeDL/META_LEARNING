from __future__ import annotations

from mvr.scripts.evaluate_cutin_inner_training_gate import summarize_records


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
