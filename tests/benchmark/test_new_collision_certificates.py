from dataclasses import replace
import json

import pytest

from benchmark import certify_collisions as certification
from benchmark.certify_collisions import new_collision_metrics
from benchmark.query import PaidExecution
from benchmark.scenarios import sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from controllers.highway.catalog import BY_ID


@pytest.fixture
def certification_worlds(tmp_path, monkeypatch):
    scenes = sample("front_brake", 3, "new", replicate=27)
    sources = (BY_ID["history_idm_mobil"], BY_ID["history_fvdm_lane"])
    targets = (BY_ID["target_idm_mobil"], BY_ID["target_fvdm_lane"])
    monkeypatch.setattr(certification, "SUTS", (*sources, *targets))
    monkeypatch.setattr(certification, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(certification, "RESULT_ROOT", tmp_path / "results")

    class TargetBank:
        def __init__(self, failed):
            self.candidates = scenes.copy()
            self.failed = set(failed)

        def __call__(self, scenario):
            index = self.candidates.index(scenario)
            return PaidExecution(QueryFeedback(scenario.scenario_id, index in self.failed,
                                               SafetyMeasurements(1, 2), 0.01), [])

    banks = {targets[0].sut_id: TargetBank((0, 1)), targets[1].sut_id: TargetBank((1, 2))}
    original_bank = certification.ExecutionBank
    original_collect = certification.collect_bank

    def load(path):
        if "targets" in path.parts:
            return banks[path.parts[path.parts.index("targets") + 1]]
        return original_bank(path)

    def collect(*args, **kwargs):
        return original_collect(*args, **kwargs, workers=1)

    monkeypatch.setattr(certification, "ExecutionBank", load)
    monkeypatch.setattr(certification, "collect_bank", collect)
    return scenes, sources, targets, banks


def test_version_new_and_historical_specific_are_distinct_real_labels():
    scenes = sample("front_brake", 2, "new")
    queries = [{"index": i, "feedback": {"collision": True}} for i in range(2)]
    certificate = {"label_provenance": "real_execution", "reference_id": "a", "condition": "new",
                   "outcomes": {scenes[0].scenario_id: {"a": False, "b": True},
                                scenes[1].scenario_id: {"a": False, "b": False}},
                   "actual_additional_source_calls": 4}
    result = new_collision_metrics(scenes, queries, certificate, ("a", "b"))
    assert result["version_new_collisions"] == 2
    assert result["historical_set_specific_collisions"] == 1
    assert result["required_additional_source_calls_if_uncached"] == 4
    certificate["label_provenance"] = "frozen_model_prediction"
    with pytest.raises(ValueError, match="Predicted historical"):
        new_collision_metrics(scenes, queries, certificate, ("a", "b"))


def test_no_history_baseline_still_uses_the_experimental_history_set_for_certification():
    scene = sample("front_brake", 1, "reuse")[0]
    certificate = {"label_provenance": "real_execution", "reference_id": "a", "condition": "reuse",
                   "outcomes": {scene.scenario_id: {"a": False, "b": False}},
                   "actual_additional_source_calls": 0}
    queries = [{"index": 0, "feedback": {"collision": True}}]
    with pytest.raises(ValueError, match="available historical set"):
        new_collision_metrics([scene], queries, certificate, ())
    result = new_collision_metrics([scene], queries, certificate, ("a", "b"))
    assert result["historical_set_specific_collisions"] == 1
    assert result["required_additional_source_calls_if_uncached"] == 0


def test_overlapping_target_failures_share_real_source_calls_and_resume_without_duplicates(certification_worlds):
    scenes, sources, targets, _ = certification_worlds
    shared = certification.shared_source_banks("new", "front_brake", 81001)
    assert set(shared) == {s.sut_id for s in sources}
    assert all(bank.candidates == scenes for bank in shared.values())
    assert sum(bank.manifest["attempted_calls"] for bank in shared.values()) == 6
    reports = [certification.certify_bank(t.sut_id, "new", "front_brake", 81001, shared_sources=shared)
               for t in targets]
    assert set(reports[0]["outcomes"]) == {s.scenario_id for s in scenes[:2]}
    assert set(reports[1]["outcomes"]) == {s.scenario_id for s in scenes[1:]}
    assert reports[0]["source_bank_paths"] == reports[1]["source_bank_paths"]
    for report in reports:
        assert report["source_bank_scope"] == "shared_across_targets"
        assert "not additive" in report["cost_accounting"]
        for scene in scenes:
            if scene.scenario_id in report["outcomes"]:
                assert report["outcomes"][scene.scenario_id] == {
                    source.sut_id: shared[source.sut_id](scene).feedback.collision for source in sources}
    resumed = certification.shared_source_banks("new", "front_brake", 81001)
    assert sum(bank.manifest["attempted_calls"] for bank in resumed.values()) == 6
    for bank in resumed.values():
        calls = [json.loads(line) for line in (bank.root / "calls.jsonl").read_text().splitlines()]
        assert sum(row["status"] == "scheduled" for row in calls) == 3


def test_source_certification_refuses_sharing_different_target_worlds(certification_worlds, monkeypatch):
    scenes, _, targets, banks = certification_worlds
    banks[targets[1].sut_id].candidates[0] = replace(
        scenes[0], parameters={**scenes[0].parameters, "gap_m": 65})

    def forbidden(*args, **kwargs):
        raise AssertionError("Source calls must wait for matching target-world validation")

    monkeypatch.setattr(certification, "collect_bank", forbidden)
    with pytest.raises(ValueError, match="identical target candidate worlds"):
        certification.shared_source_banks("new", "front_brake", 81001)


def test_no_target_collisions_require_no_additional_source_calls(certification_worlds, monkeypatch):
    _, _, _, banks = certification_worlds
    for bank in banks.values():
        bank.failed.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("A passing pool has no collision labels to certify")

    monkeypatch.setattr(certification, "collect_bank", forbidden)
    assert certification.shared_source_banks("new", "front_brake", 81001) == {}
