import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.highway.catalog import BY_ID
from controllers.highway.control import Control
from experiments import prepare_joint as preparation
from experiments.history import calculation_view, load_training_archives
from experiments.rss import measurement_contract
from experiments.scenarios import candidate_pool, difficult, sample, validate
from experiments.storage import save_archive, write_json
from simulator.highway.collision_env import execute
from testing_methods.common.response import ResponseArchive, archive_fingerprints


class Brake:
    def act(self, world):
        return Control(-8, 0)


def bank_fixture(root, sut, scenes, *, version="recorded_fixture"):
    root = Path(root)
    (root / "executions").mkdir(parents=True, exist_ok=True)
    contract = {"fixture": "short real trajectories", "object": {**asdict(sut), "version": version}}
    write_json(
        root / "manifest.json",
        {
            "format": "collision-execution-bank",
            "status": "complete",
            "sut_id": sut.sut_id,
            "contract": contract,
            "environment_seed": 81001,
            "scenarios": [asdict(s) for s in scenes],
        },
    )
    calls, measurements = [], []
    for index, scene in enumerate(scenes):
        execution = execute(Brake(), scene, duration_s=0.2)
        measurements.append(execution.feedback)
        payload = {
            "call": index + 1,
            "scenario": asdict(scene),
            "feedback": execution.feedback.to_dict(),
            "trajectory": execution.trajectory,
        }
        (root / f"executions/{index:06d}.json.gz").write_bytes(
            gzip.compress(json.dumps(payload).encode())
        )
        calls.append({"call": index + 1, "index": index, "status": "complete"})
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls), encoding="utf-8")
    archive = ResponseArchive(
        sut.sut_id,
        sut.family,
        sut.lineage,
        sut.role,
        scenes,
        np.array([m.collision for m in measurements]),
        np.array([[m.safety.min_clearance_m, m.safety.min_ttc_s] for m in measurements]),
        provenance={"qualified": True, "bank": str(root), "contract": contract},
    )
    return archive


@pytest.mark.parametrize("kind", preparation.TEST_SCENARIOS)
def test_sobol_supplement_keeps_difficult_points_and_excludes_only_existing_coordinates(kind):
    original = sample(kind, 512, "history") if kind != "car_following" else []
    excluded = {s.coordinate() for s in original}
    added = preparation.supplement_coordinates(kind, preparation.ADDITIONS[kind], excluded=excluded)
    assert len(added) == preparation.ADDITIONS[kind]
    assert len({s.coordinate() for s in added}) == len(added)
    assert all(s.coordinate() not in excluded for s in added)
    assert any(difficult(s) for s in added)
    for scene in added:
        validate(scene)
    assert added == preparation.supplement_coordinates(kind, len(added), excluded=excluded)
    assert not {s.coordinate() for s in added}.intersection(
        s.coordinate() for s in candidate_pool(kind)
    )
    archive = ResponseArchive(
        "source",
        "rule",
        "fixture",
        "history",
        added,
        np.zeros(len(added), dtype=bool),
        np.ones((len(added), 2)),
        rss_margin=np.zeros(len(added)),
        rss_contract=measurement_contract(),
    )
    assert len(archive.scenarios) == len(added)


def test_prepare_merges_separate_batches_and_preserves_original_evidence(tmp_path, monkeypatch):
    raw, destination = tmp_path / "raw", tmp_path / "joint"
    monkeypatch.setattr(preparation, "DATA_ROOT", raw)
    monkeypatch.setattr(preparation, "require_qualified", lambda sut: None)
    systems = [BY_ID["history_idm_mobil"], BY_ID["history_fvdm_lane"]]
    originals = {}
    for sut in systems:
        archive = bank_fixture(
            raw / "physical" / sut.sut_id, sut, sample("front_brake", 2, "history")
        )
        path = raw / "history/front_brake" / f"{sut.sut_id}.npz"
        save_archive(path, archive, provenance=archive.provenance)
        originals[path] = path.read_bytes()
    collection_coordinates = []
    calls = []

    def collect(sut_id, scenes, root, **kwargs):
        collection_coordinates.append((scenes[0].kind, sut_id, scenes))
        if not (Path(root) / "manifest.json").exists():
            calls.extend((sut_id, s.scenario_id) for s in scenes)
            bank_fixture(root, BY_ID[sut_id], scenes, version="supplement_fixture")
        return SimpleNamespace(root=root)

    monkeypatch.setattr(preparation, "collect_bank", collect)
    options = dict(
        kinds=("car_following", "front_brake"),
        systems=systems,
        additions={"car_following": 2, "front_brake": 2},
        recorded_count=2,
    )
    result = preparation.prepare_dataset(destination, **options)
    assert result["recorded_records"] == 4
    assert result["supplement_records"] == 8
    assert result["total_records"] == 12
    assert not result["development_ads_used"]
    assert not (destination / "development").exists()
    assert len(calls) == 8
    assert collection_coordinates[0][2] == collection_coordinates[1][2]
    assert collection_coordinates[2][2] == collection_coordinates[3][2]
    archives = load_training_archives(destination / "manifest.json", "front_brake")
    scoped = load_training_archives(
        destination / "history/front_brake/manifest.json", "front_brake"
    )
    assert archive_fingerprints(archives) == archive_fingerprints(scoped)
    for archive in archives:
        assert len(archive.scenarios) == 4
        assert len(archive.provenance["record_ids"]) == 4
        assert {r["sut_version"] for r in archive.provenance["executions"].values()} == {
            "recorded_fixture",
            "supplement_fixture",
        }
        view = calculation_view(archive, limit=2)
        assert len(view.provenance["record_ids"]) == 2
        assert set(view.provenance["record_ids"]) <= set(archive.provenance["record_ids"])
    assert preparation.prepare_dataset(destination, **options) == result
    assert len(calls) == 8
    assert all(p.read_bytes() == contents for p, contents in originals.items())


def test_conversion_requires_actual_collision_and_completed_execution_receipts(tmp_path):
    sut = BY_ID["history_idm_mobil"]
    bank_fixture(tmp_path, sut, sample("front_brake", 2, "history"))
    first = preparation.response_from_bank(tmp_path, batch="recorded")
    assert len(first.continuous_margin) == 2
    assert all(r["rss_compute_s"] >= 0 for r in first.provenance["executions"].values())
    path = tmp_path / "executions/000000.json.gz"
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["feedback"]["collision"] = not payload["feedback"]["collision"]
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="execution evidence"):
        preparation.response_from_bank(tmp_path, batch="recorded")
    bank_fixture(tmp_path, sut, sample("front_brake", 2, "history"))
    (tmp_path / "calls.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="execution evidence"):
        preparation.response_from_bank(tmp_path, batch="recorded")


def test_batch_identity_and_repeated_records_have_explicit_handling(tmp_path):
    sut = BY_ID["history_idm_mobil"]
    bank_fixture(tmp_path, sut, sample("front_brake", 2, "history"))
    archive = preparation.response_from_bank(tmp_path, batch="recorded")
    merged = preparation.merge_batches([archive, archive])
    assert len(merged.scenarios) == 2
    assert merged.provenance["repeated_execution_ids"] == archive.provenance["record_ids"]
    with pytest.raises(ValueError, match="same historical ADS"):
        preparation.merge_batches([archive, replace(archive, source_id="different")])
    before = archive_fingerprints([merged])
    merged.provenance["contract"]["batches"][0]["contract"]["object"]["version"] = "changed"
    assert archive_fingerprints([merged]) != before
    for other in (BY_ID["independent_idm_mobil"], replace(sut, role="development")):
        with pytest.raises(ValueError, match="original historical"):
            preparation.historical_systems([other])
