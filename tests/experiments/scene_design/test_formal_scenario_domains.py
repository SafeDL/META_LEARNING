"""Integer physical domains have unique, disjoint, outcome-independent candidates."""

from itertools import product

import numpy as np
import pytest

from experiments import history_transfer_experiment as runner
from experiments.formal_scenarios import (
    FREE_BOUNDS,
    HISTORY_COUNTS,
    candidate_payload,
    coordinate_stream,
    development_coordinates,
    map_scene,
    prepare_candidates,
    verify_candidates,
)
from experiments.scenarios import features
from experiments.schema import Scenario
from experiments.storage import read_json, write_json


@pytest.mark.parametrize("kind", FREE_BOUNDS)
def test_entire_cube_corners_have_valid_physical_geometry_and_common_features(kind):
    scenes = []
    for point in product((0, 1), repeat=len(FREE_BOUNDS[kind])):
        scene, _ = map_scene(kind, point, "corner")
        scenes.append(scene)
        p = scene.parameters
        assert all(type(value) is int for value in p.values())
    values = features(scenes)
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1


@pytest.mark.parametrize("kind", FREE_BOUNDS)
def test_each_numeric_coordinate_changes_only_its_own_physical_parameter(kind):
    point = np.full(len(FREE_BOUNDS[kind]), 0.5)
    before, _ = map_scene(kind, point, "a")
    for axis, name in enumerate(FREE_BOUNDS[kind]):
        changed = point.copy()
        changed[axis] = 0
        after, _ = map_scene(kind, changed, "b")
        assert {k for k in before.parameters if before.parameters[k] != after.parameters[k]} == {
            name
        }
    for invalid in (point[:-1], point * np.nan, point + 1):
        with pytest.raises(ValueError, match="independent parameter"):
            map_scene(kind, invalid, "invalid")


@pytest.mark.parametrize("kind", FREE_BOUNDS)
def test_full_splits_have_no_shared_or_duplicate_physical_coordinates(kind):
    populations = []
    for split, count in (
        ("history", HISTORY_COUNTS[kind]),
        ("target", 2000),
    ):
        records = list(coordinate_stream(kind, split))
        coordinates = {
            (tuple(r["independent"].values()), tuple(r["scenario"]["categories"].values()))
            for r in records
        }
        assert len(records) == len(coordinates) == count
        assert development_coordinates(kind).isdisjoint(
            tuple(r["independent"].values()) for r in records
        )
        assert all(not coordinates.intersection(existing) for existing in populations)
        populations.append(coordinates)


@pytest.mark.parametrize("kind", FREE_BOUNDS)
def test_direct_preparation_is_exact_resumable_and_rejects_changed_records(kind, tmp_path):
    scenes = prepare_candidates(kind, 32, "history", tmp_path)
    assert scenes == prepare_candidates(kind, 32, "history", tmp_path)
    assert scenes == verify_candidates(tmp_path, kind, 32, "history")
    saved = read_json(tmp_path / "manifest.json")
    assert (
        saved["generated_count"] == 32
        and saved["rejected_count"] == saved["physical_executions"] == 0
    )
    assert len({s.coordinate() for s in scenes}) == 32
    target = candidate_payload(kind, 32, "target")
    assert {scene.coordinate() for scene in scenes}.isdisjoint(
        Scenario(**r["scenario"]).coordinate() for r in target["records"]
    )
    saved["records"][0]["unit_coordinate"][0] += 0.01
    write_json(tmp_path / "manifest.json", saved)
    with pytest.raises(ValueError, match="registered domain or coordinates"):
        verify_candidates(tmp_path, kind, 32, "history")


def test_collection_stops_before_any_execution_when_domain_evidence_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "require_qualified", lambda sut: "qualification-evidence")

    def missing():
        raise ValueError("missing domain evidence")

    def forbidden(*args, **kwargs):
        raise AssertionError("No physical execution may start before the collection gate")

    monkeypatch.setattr(runner, "verify_development", missing)
    monkeypatch.setattr(runner, "collect_bank", forbidden)
    state = runner.readiness(tmp_path)
    assert state["ads_ready"] and not state["collection"]["ready"] and not state["ready"]
    with pytest.raises(ValueError, match="prerequisites"):
        runner.collect(tmp_path)
    with pytest.raises(ValueError, match="missing domain evidence"):
        runner.prepare(tmp_path)


@pytest.mark.parametrize("stage", (runner.run_k_shot, runner.run_budget))
def test_collection_ready_does_not_open_unverified_method_evaluation(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(
        runner,
        "readiness",
        lambda root: {"collection": {"ready": True}, "method_evaluation": {"ready": False}},
    )
    with pytest.raises(ValueError, match="method evaluation prerequisites|deferred"):
        stage(tmp_path)


@pytest.mark.parametrize("kind", ("cut_in_interaction", "cut_out", "lane_change_rear"))
def test_replaced_domains_cannot_generate_formal_candidates(kind):
    with pytest.raises(ValueError, match="no current formal sampling registration"):
        candidate_payload(kind, 1, "history")
