"""A scene can train before other scenes finish, but never from partial sources."""

from dataclasses import asdict

import numpy as np
import pytest

from experiments import history_transfer_experiment as runner
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from experiments.storage import read_json, write_json
from testing_methods.common.response import ResponseArchive


@pytest.fixture
def completed_sources(tmp_path, monkeypatch):
    kind = "car_following"
    scenes = sample(kind, 3, "history")
    systems = runner.FORMAL_HISTORY_SUTS[:2]
    archives = {}
    monkeypatch.setattr(runner, "FORMAL_HISTORY_SUTS", systems)
    monkeypatch.setattr(runner, "scenarios_for", lambda _: (kind,))
    monkeypatch.setitem(runner.HISTORY_COUNTS, kind, 3)
    monkeypatch.setattr(runner, "registered_populations", lambda _: {("history", kind): scenes})
    monkeypatch.setattr(runner, "execution_contract", lambda sut, **_: {"sut": asdict(sut)})
    for sut in systems:
        contract = runner.execution_contract(sut)
        archive = ResponseArchive(
            sut.sut_id,
            sut.family,
            sut.lineage,
            "history",
            scenes,
            np.zeros(3, dtype=bool),
            np.zeros((3, 2)),
            provenance={"contract": contract},
            rss_margin=np.full(3, 0.5),
            rss_contract=measurement_contract(),
        )
        archives[sut.sut_id] = archive
        bank = tmp_path / "physical/history" / kind / sut.sut_id
        write_json(
            bank / "manifest.json",
            {
                "format": "collision-execution-bank",
                "status": "complete",
                "scenarios": [asdict(s) for s in scenes],
                "contract": contract,
                "environment_seed": 81001,
            },
        )
        for index in range(3):
            write_json(bank / "executions" / f"{index:06d}.json.gz", {"fixture": index})
    monkeypatch.setattr(runner, "load_archive", lambda path: archives[path.stem])
    monkeypatch.setattr(runner, "load_training_archives", lambda *_: list(archives.values()))
    return tmp_path, kind, systems, archives


def test_completed_scene_does_not_need_global_or_target_manifests(completed_sources):
    root, kind, systems, archives = completed_sources
    assert runner.scenario_history(root, kind) == list(archives.values())
    assert not (root / "data/manifest.json").exists()
    assert not (root / "physical/target").exists()
    manifest = read_json(root / "data/history" / kind / "manifest.json")
    assert [r["source_id"] for r in manifest["archives"]] == [s.sut_id for s in systems]
    assert all(r["records"] == 3 and r["physical_bank_snapshot"] for r in manifest["archives"])
    runner.scenario_history(root, kind)
    archives[systems[0].sut_id].collision[0] = True
    with pytest.raises(ValueError, match="Registered protocol changed"):
        runner.scenario_history(root, kind)


@pytest.mark.parametrize("change", ["partial", "identity", "coordinates", "contract", "missing"])
def test_scene_rejects_partial_changed_or_missing_sources(completed_sources, change, monkeypatch):
    root, kind, systems, archives = completed_sources
    sut = systems[-1]
    path = root / "physical/history" / kind / sut.sut_id / "manifest.json"
    bank = read_json(path)
    if change == "partial":
        bank["status"] = "running"
    elif change == "identity":
        archives[sut.sut_id].source_id = "unregistered"
    elif change == "coordinates":
        archives[sut.sut_id].scenarios = sample(kind, 3, "history", replicate=500)
    elif change == "contract":
        bank["contract"] = {"changed": True}
    else:

        def missing(_):
            raise FileNotFoundError("Missing historical archive")

        monkeypatch.setattr(runner, "load_archive", missing)
    write_json(path, bank)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.scenario_history(root, kind)
    assert not (root / "data/history" / kind / "manifest.json").exists()


def test_scene_rejects_changed_physical_executions(completed_sources):
    root, kind, systems, _ = completed_sources
    runner.scenario_history(root, kind)
    path = root / "physical/history" / kind / systems[0].sut_id / "executions/000000.json.gz"
    write_json(path, {"fixture": "changed physical bytes"})
    with pytest.raises(ValueError, match="Registered protocol changed"):
        runner.scenario_history(root, kind)


def test_new_training_keeps_the_fixed_mixture_model_intact(completed_sources, monkeypatch):
    root, kind, _, history = completed_sources
    old = root / "models" / kind / "historical_encoder.pt"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"fixed-mixture source model")
    calls = []
    monkeypatch.setattr(runner, "register", lambda _: None)
    monkeypatch.setattr(runner, "train", lambda records, path, **_: calls.append((records, path)))
    runner.train_scenario(kind, root)
    assert calls == [(list(history.values()), root / "models/adaptive_mixture" / kind)]
    assert old.read_bytes() == b"fixed-mixture source model"
