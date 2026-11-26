import gzip
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from controllers.highway.catalog import BY_ID
from experiments import historical_responses as preparation
from experiments.formal_scenarios import DEVELOPMENT_ROOT
from experiments.storage import read_json, write_json


def bank_fixture(root, sut):
    root = Path(root)
    (root / "executions").mkdir(parents=True, exist_ok=True)
    contract = {"fixture": "read-only archived trajectories", "object": asdict(sut)}
    payloads = [read_json(DEVELOPMENT_ROOT / sut.sut_id / f"{i:03d}.json.gz") for i in range(2)]
    write_json(
        root / "manifest.json",
        {
            "format": "collision-execution-bank",
            "status": "complete",
            "sut_id": sut.sut_id,
            "contract": contract,
            "environment_seed": 81001,
            "scenarios": [p["scenario"] for p in payloads],
        },
    )
    calls = []
    for index, archived in enumerate(payloads):
        payload = {
            "call": index + 1,
            "scenario": archived["scenario"],
            "feedback": archived["feedback"],
            "trajectory": archived["trajectory"],
        }
        (root / f"executions/{index:06d}.json.gz").write_bytes(
            gzip.compress(json.dumps(payload).encode())
        )
        calls.append({"call": index + 1, "index": index, "status": "complete"})
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls), encoding="utf-8")


def test_conversion_requires_actual_collision_and_completed_execution_receipts(tmp_path):
    sut = BY_ID["history_idm_mobil"]
    bank_fixture(tmp_path, sut)
    first = preparation.response_from_bank(tmp_path, batch="recorded")
    assert len(first.continuous_margin) == 2
    assert all(r["rss_compute_s"] >= 0 for r in first.provenance["executions"].values())
    path = tmp_path / "executions/000000.json.gz"
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["feedback"]["collision"] = not payload["feedback"]["collision"]
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="execution evidence"):
        preparation.response_from_bank(tmp_path, batch="recorded")
    bank_fixture(tmp_path, sut)
    (tmp_path / "calls.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="execution evidence"):
        preparation.response_from_bank(tmp_path, batch="recorded")
