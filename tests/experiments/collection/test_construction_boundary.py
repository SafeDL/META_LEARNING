import json

import numpy as np

from controllers.highway.catalog import BY_ID, CURRENT_SUTS
from experiments import qualification
from experiments.executions import execution_contract
from experiments.scenarios import sample
from testing_methods.common.response import ResponseArchive, archive_fingerprints


def test_qualification_prepares_current_individual_records_without_retired_aggregates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualification, "QUALIFICATION_ROOT", tmp_path)
    monkeypatch.setattr(qualification, "create", lambda name: name)
    evaluated = []

    def qualify(factory, *, save_root):
        evaluated.append(factory())
        assert save_root.name == evaluated[-1]
        save_root.mkdir()
        return {"qualified": True, "courses": []}

    monkeypatch.setattr(qualification, "qualify", qualify)
    rows = qualification.main()
    expected = {sut.sut_id for sut in CURRENT_SUTS if sut.family != "learning"}
    assert set(evaluated) == expected and len(rows) == 12
    assert {path.name for path in tmp_path.iterdir()} == expected
    for name in expected:
        report = json.loads((tmp_path / name / "qualification.json").read_text(encoding="utf-8"))
        assert report["qualified"] and report["parameters"] == BY_ID[name].parameters


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
