import json
from dataclasses import asdict

import numpy as np
import pytest

from controllers.highway import train
from controllers.highway.catalog import BY_ID
from experiments import construction
from experiments.executions import execution_contract
from experiments.protocol import DEFAULT
from experiments.scenarios import sample
from experiments.storage import save_archive
from testing_methods.common.response import ResponseArchive, archive_fingerprints


def write_archive(root, sut, *, changed_version=False, count=DEFAULT.candidate_count):
    scenes = sample("front_brake", count, sut.role)
    contract = execution_contract(sut)
    if changed_version:
        contract["object"]["version"] = "retired_construction"
    archive = ResponseArchive(
        sut.sut_id,
        sut.family,
        sut.lineage,
        sut.role,
        scenes,
        np.zeros(count, dtype=bool),
        np.ones((count, 2)),
    )
    save_archive(
        root / sut.role / "front_brake" / f"{sut.sut_id}.npz",
        archive,
        provenance={"qualified": True, "contract": contract},
    )


@pytest.mark.parametrize("stale", ["version", "coordinates", None])
def test_construction_checks_actual_registered_version_and_coordinate_pool(
    tmp_path, monkeypatch, stale
):
    history, development = BY_ID["history_idm_mobil"], BY_ID["development_idm_mobil"]
    monkeypatch.setattr(construction, "SUTS", [history, development])
    monkeypatch.setattr(construction, "TEST_RECORDS_ROOT", tmp_path)
    write_archive(
        tmp_path,
        history,
        changed_version=stale == "version",
        count=16 if stale == "coordinates" else DEFAULT.candidate_count,
    )
    write_archive(tmp_path, development)
    if stale:
        with pytest.raises(ValueError, match="version|coordinates"):
            construction.load_construction("front_brake")
    else:
        historical, dev = construction.load_construction("front_brake")
        assert [a.source_id for a in historical + dev] == [history.sut_id, development.sut_id]


def test_training_freeze_reads_registered_runs_and_json_serializable_descriptors(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(train, "CHECKPOINT_ROOT", tmp_path)
    for name in ("history_ppo", "history_dqn", "development_ppo"):
        sut = BY_ID[name]
        root = tmp_path / sut.training_run
        root.mkdir()
        evidence = {
            "status": "qualified",
            "configuration": train.TRAINING,
            "source_contract": train.training_contract(),
            "object": asdict(sut),
        }
        (root / "training.json").write_text(json.dumps(evidence), encoding="utf-8")
    train.freeze_training()
    frozen = json.loads((tmp_path / "training_protocol.json").read_text())
    assert frozen["qualified_runs_before_targets"] == [
        BY_ID[name].training_run for name in ("history_ppo", "history_dqn", "development_ppo")
    ]
    assert frozen["configuration"] == train.TRAINING
    for sut in BY_ID.values():
        assert json.loads(json.dumps(asdict(sut))) == asdict(sut)


def test_response_fingerprints_include_the_executed_controller_version():
    sut = BY_ID["history_idm_mobil"]
    archive = ResponseArchive(
        sut.sut_id,
        sut.family,
        sut.lineage,
        sut.role,
        sample("front_brake", 16, "history"),
        np.zeros(16, dtype=bool),
        np.ones((16, 2)),
        provenance={"contract": execution_contract(sut)},
    )
    before = archive_fingerprints([archive])
    archive.provenance["contract"]["object"]["version"] = "different_executed_version"
    assert archive_fingerprints([archive]) != before
    assert not archive.collision.any()
