import gzip
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from controllers.highway.control import Control
from experiments.history import build_source, calculation_view, completed_archives, finalize_task
from experiments.query import QueryLedger
from experiments.rss import measurement_contract, trajectory_margin
from experiments.scenarios import DOMAINS, sample
from experiments.storage import load_archive, save_archive
from simulator.highway.collision_env import execute
from testing_methods.history_transfer.model import JointResponseKernel


class Brake:
    def act(self, world):
        return Control(-8, 0)


def ledger(n=8):
    scenes = sample("car_following", n, "new", replicate=3)
    result = QueryLedger(
        scenes, n, lambda s: execute(Brake(), s, duration_s=0.4), joint_feedback=True
    )
    for i in range(n):
        result.query(i, purpose="initial" if i == 0 else "target")
    return result


def finalize(path, paid, **changes):
    options = dict(
        task_id="task",
        sut_id="new_ads",
        sut_version="original_weights",
        family="rule",
        lineage="idm",
        environment_seed=81001,
        completed_order=1,
        branch_id="branch",
    )
    options.update(changes)
    return finalize_task(path, paid, **options)


@pytest.mark.parametrize("n", [1, 8, 32, 128])
def test_task_archives_only_paid_coordinates_and_uses_temporal_branch_admission(tmp_path, n):
    paid = ledger(n)
    path = finalize(tmp_path / "task.json.gz", paid)
    assert finalize(path, paid) == path
    saved = json.loads(gzip.decompress(path.read_bytes()))
    assert all("trajectory" not in r["feedback"] for r in saved["attempts"])
    for execution, original in zip(saved["executions"], paid.trajectories, strict=True):
        assert "trajectory" not in execution["feedback"]
        assert execution["trajectory"] == original
    assert completed_archives([path], before_order=1, branch_id="branch") == []
    assert completed_archives([path], before_order=2, branch_id="other") == []
    assert (
        completed_archives(
            [path],
            before_order=2,
            branch_id="branch",
            current_system=("new_ads", "original_weights"),
        )
        == []
    )
    assert completed_archives([path], before_order=2, branch_id="branch", held_family="rule") == []
    dev_path = finalize(
        tmp_path / "dev.json.gz", paid, task_id="development", original_role="development"
    )
    assert completed_archives([dev_path], before_order=2, branch_id="branch") == []
    archives = completed_archives([path], before_order=2, branch_id="branch")
    assert len(archives) == 1 and archives[0].role == "target"
    assert len(archives[0].scenarios) == n
    assert archives[0].provenance["original_role"] == "target"
    assert archives[0].scenarios == list(paid.candidates)
    archive_path = tmp_path / "source.npz"
    save_archive(archive_path, archives[0], provenance=archives[0].provenance)
    restored = load_archive(archive_path)
    np.testing.assert_array_equal(restored.continuous_margin, archives[0].continuous_margin)


def test_query_rss_is_derived_only_after_payment_and_records_its_cost():
    paid = ledger(2)
    for record, trajectory in zip(paid.records, paid.trajectories):
        assert record["rss_compute_s"] >= 0
        assert record["feedback"]["rss_contract"] == measurement_contract()
        assert record["feedback"]["trajectory"] == trajectory
        assert (
            record["feedback"]["safety"]["rss_margin"]
            == trajectory_margin(trajectory)["minimum_margin"]
        )
    with pytest.raises(ValueError, match="budget exhausted"):
        paid.query(0, purpose="repeat")


def test_completed_task_retains_its_registered_parameter_definition(tmp_path):
    paid = ledger(2)
    definition = DOMAINS["car_following"]
    path = finalize(tmp_path / "defined_task.json.gz", paid, domain=definition)
    restored = completed_archives([path], before_order=2, branch_id="branch")[0]
    assert restored.domain.to_dict() == definition.to_dict()
    np.testing.assert_array_equal(restored.x, definition.features(restored.scenarios))


def test_failed_attempts_are_audited_and_unqueried_bank_results_are_not_archived(tmp_path):
    scenes = sample("car_following", 4, "new", replicate=3)

    def executor(scene):
        if scene.scenario_id == scenes[1].scenario_id:
            raise RuntimeError("simulator failed")
        return execute(Brake(), scene, duration_s=0.2)

    paid = QueryLedger(scenes, 3, executor, joint_feedback=True)
    paid.query(0)
    with pytest.raises(RuntimeError):
        paid.query(1)
    paid.query(2)
    path = finalize(tmp_path / "task.json.gz", paid)
    payload = json.loads(gzip.decompress(path.read_bytes()))
    assert [r["status"] for r in payload["attempts"]] == ["complete", "failed", "complete"]
    assert len(payload["executions"]) == 2
    archive = completed_archives([path], before_order=2, branch_id="branch")[0]
    assert archive.scenarios == [scenes[0], scenes[2]]


def test_source_cache_binds_version_coordinates_rss_and_archive_snapshot(tmp_path):
    torch.set_num_threads(1)
    path = finalize(tmp_path / "task.json.gz", ledger(8))
    archives = completed_archives([path], before_order=2, branch_id="branch")
    model = JointResponseKernel(archives[0].x.shape[1]).double()
    candidates = sample("car_following", 12, "new", replicate=4)
    cache = tmp_path / "source.npz"
    before = {k: v.clone() for k, v in model.state_dict().items()}
    first = build_source(model, archives, candidates, cache_path=cache, limit=4, batch_size=3)
    second = build_source(model, archives, candidates, cache_path=cache, limit=4)
    np.testing.assert_array_equal(first["means"], second["means"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    changed = [
        replace(candidates[0], parameters={**candidates[0].parameters, "gap_m": 60}),
        *candidates[1:],
    ]
    with pytest.raises(ValueError, match="snapshot"):
        build_source(model, archives, changed, cache_path=cache, limit=4)
    with torch.no_grad():
        model.mean.add_(0.01)
    with pytest.raises(ValueError, match="snapshot"):
        build_source(model, archives, candidates, cache_path=cache, limit=4)


def test_bounded_view_keeps_original_archive_and_order(tmp_path):
    path = finalize(tmp_path / "task.json.gz", ledger(32))
    archive = completed_archives([path], before_order=2, branch_id="branch")[0]
    view = calculation_view(archive, limit=7)
    assert len(archive.scenarios) == 32 and len(view.scenarios) == 7
    assert view.provenance["calculation_indices"] == sorted(view.provenance["calculation_indices"])
    assert view.provenance["record_ids"] == [
        archive.provenance["record_ids"][i] for i in view.provenance["calculation_indices"]
    ]


def test_explicit_records_fix_the_rss_control_view_and_invalidate_other_cache(tmp_path):
    path = finalize(tmp_path / "task.json.gz", ledger(8))
    archive = completed_archives([path], before_order=2, branch_id="branch")[0]
    ids = archive.provenance["record_ids"][1::2]
    views = [calculation_view(archive, limit=4, use_rss=r, record_ids=ids) for r in (True, False)]
    assert views[0].provenance == views[1].provenance
    assert views[0].scenarios == views[1].scenarios
    for bad in ([ids[0], ids[0]], ["absent"], []):
        with pytest.raises(ValueError, match="record"):
            calculation_view(archive, record_ids=bad)
    with pytest.raises(ValueError, match="limit"):
        calculation_view(archive, limit=3, record_ids=ids)
    model = JointResponseKernel(archive.x.shape[1]).double()
    cache = tmp_path / "fixed.npz"
    mapping = {archive.source_id: ids}
    source = build_source(
        model, [archive], archive.scenarios, cache_path=cache, view_record_ids=mapping
    )
    assert source["specification"]["view_record_ids"] == mapping
    with pytest.raises(ValueError, match="snapshot"):
        build_source(
            model,
            [archive],
            archive.scenarios,
            cache_path=cache,
            view_record_ids={archive.source_id: ids[::-1]},
        )
    with pytest.raises(ValueError, match="every admitted source"):
        build_source(model, [archive], archive.scenarios, view_record_ids={})


def test_identical_source_snapshots_share_one_cache_across_runs(tmp_path, monkeypatch):
    from experiments import history as workflow

    paid_path = finalize(tmp_path / "task.json.gz", ledger(8))
    archives = completed_archives([paid_path], before_order=2, branch_id="branch")
    model = JointResponseKernel(archives[0].x.shape[1]).double()
    candidates = sample("car_following", 12, "new", replicate=4)
    first = build_source(model, archives, candidates, cache_directory=tmp_path / "cache")

    def unexpected(*args, **kwargs):
        raise AssertionError("Identical frozen source predictions must be reused")

    monkeypatch.setattr(workflow, "historical_prior", unexpected)
    second = build_source(model, archives, candidates, cache_directory=tmp_path / "cache")
    np.testing.assert_array_equal(first["means"], second["means"])
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
