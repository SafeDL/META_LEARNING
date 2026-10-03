from __future__ import annotations

import numpy as np

from replications.benchmark import _metric_rows, _summaries


def test_discovery_metrics_use_the_complete_candidate_pool() -> None:
    failures = np.asarray([True, False, True, False])
    rows = _metric_rows(
        "method", "protocol", "target", 0, [0, 1, 2, 3], failures, [2, 4], "none"
    )
    recall = {
        int(row["budget"]): float(row["value"])
        for row in rows
        if row["metric"] == "recall"
    }
    assert recall == {2: 0.5, 4: 1.0}


def test_summary_keeps_tasks_and_protocols_separate() -> None:
    records = [
        {
            "task": "failure_discovery",
            "protocol": "static",
            "method": "m",
            "budget": 5,
            "metric": "recall",
            "value": value,
        }
        for value in (0.2, 0.4)
    ]
    summary = _summaries(records)
    assert len(summary) == 1
    assert np.isclose(float(summary[0]["mean"]), 0.3)
    assert summary[0]["records"] == 2

