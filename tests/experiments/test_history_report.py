"""Historical diagnostics distinguish actual collisions, continuous margins and initial ties."""

import gzip
import json
from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image

from experiments import report_history
from experiments.rss import measurement_contract, trajectory_margin
from experiments.schema import Scenario
from experiments.storage import save_archive, write_json
from testing_methods.common.response import ResponseArchive, archive_fingerprints


def fixture_archive(root):
    records, scenes, labels, margins = {}, [], [], []
    for index, (start, end, collision) in enumerate(
        ((100, 100, False), (50, 30, False), (20, 0, True))
    ):
        scene = Scenario(
            f"case-{index}",
            "car_following",
            {"ego_speed_mps": 20, "lead_speed_mps": 20, "gap_m": 10 + index},
        )
        ego = dict(actor_id="ego", x=0, y=0, speed=20, heading=0, length=5, width=2)
        trace = [
            {
                "time_s": time,
                "collision": collision and time > 0,
                "actors": [ego, {**ego, "actor_id": "lead", "x": gap}],
            }
            for time, gap in ((0, start), (0.05, end))
        ]
        relative = f"{index}.json.gz"
        payload = {
            "call": index + 1,
            "scenario": asdict(scene),
            "feedback": {"scenario_id": scene.scenario_id, "collision": collision},
            "trajectory": trace,
        }
        (root / relative).write_bytes(gzip.compress(json.dumps(payload).encode()))
        record_id = f"record-{index}"
        records[record_id] = {
            "bank": str(root),
            "execution": relative,
            "call": index + 1,
            "original_role": "history",
            "scenario_id": scene.scenario_id,
        }
        scenes.append(scene)
        labels.append(collision)
        margins.append(trajectory_margin(trace)["minimum_margin"])
    return ResponseArchive(
        "history_fixture",
        "rule",
        "fixture",
        "history",
        scenes,
        np.array(labels),
        np.ones((3, 2)),
        provenance={
            "contract": {"batches": [{"bank": str(root), "batch": "supplement"}]},
            "record_ids": list(records),
            "executions": records,
        },
        rss_margin=np.array(margins),
        rss_contract=measurement_contract(),
    )


def test_full_report_uses_actual_trajectories_and_keeps_undefined_groups(tmp_path):
    archive = fixture_archive(tmp_path)
    path = tmp_path / "archive.npz"
    save_archive(path, archive, provenance=archive.provenance)
    original = path.read_bytes()
    write_json(
        tmp_path / "manifest.json",
        {
            "rss_contract": measurement_contract(),
            "total_records": 3,
            "archives": [
                {
                    "source_id": archive.source_id,
                    "role": "history",
                    "scenario": "car_following",
                    "records": 3,
                    "path": path.name,
                    "fingerprint": archive_fingerprints([archive])[archive.source_id],
                }
            ],
        },
    )
    summary = report_history.report(tmp_path / "manifest.json", tmp_path / "report")
    stats = summary["scenarios"][0]
    assert stats["all"]["records"] == 3
    assert stats["collision"]["records"] == 1
    assert stats["safe"]["negative_margin_records"] == 1
    assert stats["all"]["minimum_at_initial_records"] == 1
    assert stats["safe"]["minimum_at_initial_fraction"] == 0.5
    assert path.read_bytes() == original
    assert summary["new_simulations"] == summary["method_queries"] == 0
    for name in summary["figures"]:
        with Image.open(tmp_path / "report" / name) as image:
            assert image.info["dpi"] == pytest.approx((600, 600), abs=0.1)
            image.verify()
    empty = report_history.describe([])
    assert empty["records"] == 0
    assert empty["quantiles"] is empty["minimum_at_initial_fraction"] is None


@pytest.mark.parametrize("field", ["rss", "collision", "call", "scenario"])
def test_diagnostic_rejects_mismatched_execution_evidence(tmp_path, field):
    archive = fixture_archive(tmp_path)
    if field == "rss":
        archive.rss_margin[0] += 0.1
    else:
        path = tmp_path / "0.json.gz"
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if field == "collision":
            payload["trajectory"][0]["collision"] = True
        elif field == "call":
            payload["call"] += 10
        else:
            payload["scenario"]["scenario_id"] = "wrong"
        path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="differs"):
        report_history.executed_rows(archive)


def test_grouping_keeps_each_source_and_batch_separate():
    rows = [
        {
            "scenario": "car_following",
            "source_id": source,
            "batch": batch,
            "rss_margin": margin,
            "collision": collision,
            "minimum_at_initial": initial,
        }
        for source, batch, margin, collision, initial in (
            ("a", "recorded", -0.5, False, True),
            ("a", "supplement", -0.9, True, False),
            ("b", "supplement", 0.5, False, True),
        )
    ]
    groups = report_history.grouped_statistics(rows, ("source_id", "batch"))
    assert len(groups) == 3
    assert all(group["all"]["records"] == 1 for group in groups)
    assert groups[0]["collision"]["records"] == 0
    assert groups[2]["safe"]["negative_margin_fraction"] == 0
