"""Historical collection must be usable before opening formal target executions."""

from types import SimpleNamespace

import pytest

from experiments import history_transfer_experiment as runner
from experiments.scenarios import sample
from experiments.storage import read_json, write_json


@pytest.mark.parametrize(
    "history_only,target_only,expected",
    [
        (True, False, ["history"]),
        (False, False, ["history", "target"]),
        (False, True, ["target"]),
    ],
)
def test_staged_collection_preserves_complete_history_manifest(
    tmp_path, monkeypatch, history_only, target_only, expected
):
    calls = []
    scene = sample("car_following", 1, "history")[0]
    hist, target = runner.FORMAL_HISTORY_SUTS[0], runner.FORMAL_TARGET_SUTS[0]
    monkeypatch.setattr(runner, "FORMAL_HISTORY_SUTS", (hist,))
    monkeypatch.setattr(runner, "FORMAL_TARGET_SUTS", (target,))
    monkeypatch.setattr(runner, "FORMAL_SUTS", (hist, target))
    monkeypatch.setattr(runner, "scenarios_for", lambda _: ("car_following",))
    monkeypatch.setitem(runner.HISTORY_COUNTS, "car_following", 1)
    monkeypatch.setattr(runner, "readiness", lambda _: {"collection": {"ready": True}})
    monkeypatch.setattr(
        runner,
        "registered_populations",
        lambda _: {("history", "car_following"): [scene], ("target", "car_following"): [scene]},
    )

    def bank(name, scenes, path, **kwargs):
        calls.append(path.relative_to(tmp_path / "physical").parts[0])
        assert kwargs["current_registration"]
        return SimpleNamespace(root=path)

    monkeypatch.setattr(runner, "collect_bank", bank)
    archive = SimpleNamespace(scenarios=[scene])
    monkeypatch.setattr(runner, "response_from_bank", lambda *_, **__: archive)
    monkeypatch.setattr(runner, "save_verified", lambda *_: archive)
    monkeypatch.setattr(
        runner, "archive_fingerprints", lambda _: {hist.sut_id: "saved response identity"}
    )
    runner.collect(tmp_path, history_only=history_only, target_only=target_only)
    assert calls == expected
    if target_only:
        assert not (tmp_path / "data/manifest.json").exists()
        prior = {"status": "complete", "archives": [{"historical": "preserve"}]}
        write_json(tmp_path / "data/manifest.json", prior)
        runner.collect(tmp_path, target_only=True)
        assert read_json(tmp_path / "data/manifest.json") == prior
        assert read_json(tmp_path / "reports/target_collection.json")["status"] == "complete"
        return
    manifest = read_json(tmp_path / "data/manifest.json")
    assert manifest["status"] == "complete"
    assert [a["source_id"] for a in manifest["archives"]] == [hist.sut_id]


def test_collection_rejects_conflicting_parts_before_any_execution(tmp_path):
    with pytest.raises(ValueError, match="either historical or target-only"):
        runner.collect(tmp_path, history_only=True, target_only=True)
    assert not (tmp_path / "physical").exists()
