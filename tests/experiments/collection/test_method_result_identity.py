"""Changed implementations, coordinates or truth must invalidate method caches."""

import gzip
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import history as historical
from experiments import history_transfer_experiment as runner
from experiments import transfer_execution as execution
from experiments.executions import ExecutionBank
from experiments.k_shot import split_candidates
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from experiments.storage import write_json
from experiments.transfer_settings import native_settings
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_bank_snapshot_changes_after_a_record_or_coordinate_changes(tmp_path):
    scene = sample("car_following", 1, "new")[0]
    manifest = {
        "format": "collision-execution-bank",
        "status": "complete",
        "scenarios": [asdict(scene)],
    }
    write_json(tmp_path / "manifest.json", manifest)
    path = tmp_path / "executions/000000.json.gz"
    path.parent.mkdir()
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"test_response": False}, stream)
    first = ExecutionBank(tmp_path).snapshot_fingerprint
    assert ExecutionBank(tmp_path).snapshot_fingerprint == first
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"test_response": True}, stream)
    changed = ExecutionBank(tmp_path).snapshot_fingerprint
    assert changed != first
    manifest["scenarios"][0]["parameters"]["gap_m"] += 0.01
    write_json(tmp_path / "manifest.json", manifest)
    assert ExecutionBank(tmp_path).snapshot_fingerprint != changed


@pytest.mark.parametrize("changed", ["implementation", "model_construction"])
def test_source_cache_checks_more_than_unchanged_model_weights(tmp_path, monkeypatch, changed):
    scenes = sample("car_following", 4, "history", replicate=502)
    archive = ResponseArchive(
        "source",
        "rule",
        "idm",
        "history",
        scenes,
        np.array([False, True, False, False]),
        np.ones((4, 2)),
        provenance={"record_ids": [f"record:{i}" for i in range(4)]},
        rss_margin=np.linspace(-0.5, 0.3, 4),
        rss_contract=measurement_contract(),
    )
    model = SharedPrivateKernel(archive.x.shape[1], deep=False).double()
    model.construction_specification = {
        "scenario": "car_following",
        "config": {"historical_rss": True},
        "parameter_definition": None,
        "training_protocol": "original",
    }
    monkeypatch.setattr(historical, "method_contract", lambda _: {"implementation": "original"})
    path = tmp_path / "source.npz"
    original_weights = historical.model_version(model)
    historical.build_source(model, [archive], scenes, cache_path=path)
    if changed == "implementation":
        monkeypatch.setattr(historical, "method_contract", lambda _: {"implementation": "changed"})
    else:
        model.construction_specification = {
            **model.construction_specification,
            "training_protocol": "changed",
        }
    assert historical.model_version(model) == original_weights
    with pytest.raises(ValueError, match="snapshot"):
        historical.build_source(model, [archive], scenes, cache_path=path)


@pytest.mark.parametrize("changed", ["code", "coordinates", "response_bank", "steps", "prior_mean"])
def test_prediction_rejects_changed_inputs_before_querying(tmp_path, monkeypatch, changed):
    scenes = sample("car_following", 8, "new", replicate=501)
    bank = SimpleNamespace(
        manifest={"sut_id": "development", "contract": {}},
        candidates=scenes,
        snapshot_fingerprint="original truth",
    )
    settings = native_settings()
    monkeypatch.setattr(runner, "method_contract", lambda _: {"implementation": "original"})
    case = execution.prediction_case(
        bank, split_candidates(scenes, 11, budget=2), 11, "target_only",
        (None, None), None, None, settings=settings, checkpoints=(0, 2),
        limits={"wall_s": 120, "child_memory_bytes": 2**30},
    )
    if changed == "code":
        monkeypatch.setattr(runner, "method_contract", lambda _: {"implementation": "changed"})
    elif changed == "coordinates":
        bank.candidates = [
            replace(
                scenes[0],
                parameters={**scenes[0].parameters, "gap_m": scenes[0].parameters["gap_m"] + 0.01},
            ),
            *scenes[1:],
        ]
    elif changed == "response_bank":
        bank.snapshot_fingerprint = "changed truth"
    elif changed == "steps":
        settings = replace(
            settings, online=replace(settings.online, raw=replace(settings.online.raw, steps=20))
        )
    else:
        settings = replace(settings, online=replace(settings.online, raw_initial_mean=-1.2))
    monkeypatch.setattr(
        execution, "_execute", lambda *a, **kw: pytest.fail("Changed inputs cannot reach querying")
    )
    with pytest.raises(ValueError, match="pre-query task declaration"):
        execution.run_prediction(
            tmp_path, case, bank, (None, None), None, None,
            settings=settings, paid_execution=None,
        )
