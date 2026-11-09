import copy
import os
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from controllers.highway.control import Control
from experiments import parallel_comparisons as parallel
from experiments import pilot
from experiments.joint_validation import Config
from experiments.rss import measurement_contract
from experiments.scenarios import candidate_pool
from experiments.storage import read_json, save_archive, write_json
from simulator.highway.collision_env import execute
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer.model import JointResponseKernel

REDUCED = {
    "av_bo_rf_ei": dict(jackknife_folds=2, tuning_trials=1, n_estimators=4),
    "samota": dict(rbf_epochs=2),
    "scenario_fuzz": dict(cycles=1, mutants=4, executions=1, filter_attempts=1),
    "paired_system_active_evaluation": dict(
        training_epochs=2, mc_draws=4, shortlist_size=4, local_samples=8, paired_samples=2
    ),
    "indago": dict(neighborhood_size=3, max_neighborhoods=2, max_stagnation=2),
}


class Brake:
    def act(self, world):
        return Control(-8, 0)


class Bank:
    def __init__(self, scenes, root):
        self.root = root
        self.candidates = scenes
        self.manifest = {"contract": {"engineering_fixture": True}, "environment_seed": 81001}
        self.records = {
            scene.scenario_id: execute(Brake(), scene, duration_s=0.2) for scene in scenes
        }

    def __call__(self, scene):
        return self.records[scene.scenario_id]

    def execution_path(self, index):
        path = self.root / f"{index:06d}.json.gz"
        scene = self.candidates[index]
        write_json(path, {"scenario": asdict(scene), "trajectory": self(scene).trajectory})
        return path


def engineering_worker(root, sut, suite, config, dataset_manifest):
    """Exercise the unchanged selectors in fresh processes, with small engineering budgets."""
    from experiments import joint_validation
    from testing_methods.baselines.scenario_fuzz.graphs import graph

    torch.set_num_threads(1)
    torch.manual_seed(1301)
    for name, options in REDUCED.items():
        module = pilot.BASELINES[name]
        module.DEFAULT = replace(module.DEFAULT, **options)
    models = tuple(JointResponseKernel(3).double() for _ in range(2))

    def classifiers(history, unused_destination):
        fuzz = pilot.fuzz_model.ScenarioRiskNetwork(hidden_dim=4, heads=1).eval()
        fuzz.fit_normalization([graph(scene) for scene in history[0].scenarios])
        return {"indago": pilot.indago_model.FailurePredictor(3).eval(), "scenario_fuzz": fuzz}

    def load_bank(unused_root, unused_sut, kind, task_config, *, replicate):
        return Bank(
            candidate_pool(kind, task_config.candidate_count, replicate=replicate), root / kind
        )

    if (root / "reports/summary.json").exists():

        def reject_execution(*args, **kwargs):
            raise AssertionError("A resumed comparison executed another query")

        joint_validation.run_isolated = reject_execution
    pilot.fixed_comparison(
        root,
        systems=(sut,),
        suite=suite,
        config=config,
        dataset_manifest=dataset_manifest,
        model_loader=lambda kind: copy.deepcopy(models),
        classifier_builder=classifiers,
        bank_loader=load_bank,
    )


def fake_worker(root, sut, suite, config, dataset_manifest):
    protocol = read_json(root.parent.parent / "reports/protocol.json")
    protocol["systems"] = [asdict(sut)]
    write_json(root / "reports/protocol.json", protocol)
    methods = protocol["main_methods"] + protocol["controls"]
    rows = []
    for size in protocol.get("candidate_sizes", [config.candidate_count]):
        if suite == "sensitivity":
            nested = {
                **protocol,
                "config": {**protocol["config"], "candidate_count": size},
                "candidate_counts": {kind: size for kind in config.scenarios},
            }
            write_json(root / str(size) / "reports/protocol.json", nested)
        for kind in config.scenarios:
            for seed in config.algorithm_seeds:
                for method in methods:
                    destination = (
                        root.parent.parent / "reused" / sut.sut_id / str(size) / kind / method
                    )
                    write_json(destination / "result.json", {"fixture": True})
                    row = {
                        "sut_id": sut.sut_id,
                        "scenario": kind,
                        "seed": seed,
                        "method": method,
                        "metrics": {"candidate_count": size, "queries": config.budget},
                        "result_path": os.path.relpath(destination / "result.json", root),
                        "reused_from": os.path.relpath(root.parent.parent / "reused", root),
                    }
                    if suite == "sensitivity":
                        row["candidate_count"] = size
                    rows.append(row)
    write_json(root / "reports/summary.json", {"status": "complete", "runs": rows})


def data_manifest(tmp_path):
    scenes = candidate_pool("car_following", 8, replicate=98)
    entries = []
    for index in range(2):
        archive = ResponseArchive(
            f"history_{index}",
            "rule",
            f"route_{index}",
            "history",
            scenes,
            np.arange(8) % 3 == index,
            np.ones((8, 2)),
            rss_margin=np.linspace(-0.8, 0.4, 8),
            rss_contract=measurement_contract(),
        )
        path = tmp_path / "data" / f"{archive.source_id}.npz"
        save_archive(path, archive, provenance=archive.provenance)
        entries.append(
            {
                "scenario": "car_following",
                "source_id": archive.source_id,
                "path": path.name,
                "fingerprint": archive_fingerprints([archive])[archive.source_id],
            }
        )
    manifest = tmp_path / "data/manifest.json"
    write_json(manifest, {"archives": entries})
    return manifest


@pytest.mark.parametrize("suite", ["fixed", "mechanisms", "sensitivity"])
def test_dispatch_preserves_original_entry_options(tmp_path, monkeypatch, suite):
    calls = []
    monkeypatch.setattr(parallel, "fixed_comparison", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        parallel.candidate_sensitivity, "run", lambda *args, **kwargs: calls.append(kwargs)
    )
    sut = CURRENT_INDEPENDENT_SUTS[0]
    parallel.run_system(tmp_path, sut, suite, Config(), tmp_path / "manifest.json")
    options = calls[0]
    assert options["systems"] == (sut,)
    if suite == "fixed":
        assert options["methods"] == (*pilot.MAIN_METHODS, *pilot.CONTROLS)
        assert options["reuse_root"] == parallel.OUTPUT / "pilot"
        assert options["classifier_builder"].keywords == {"train_missing": False}
    elif suite == "mechanisms":
        assert options["methods"] == ("joint", *pilot.MECHANISMS)
        assert options["reuse_root"] == parallel.OUTPUT / "fixed"
        assert (
            options["historical_collision_loader"]
            is parallel.mechanisms.historical_collision_models
        )
    else:
        assert options["dataset_manifest"] == tmp_path / "manifest.json"


def test_rebased_rows_keep_one_task_each_in_registered_order(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "OUTPUT", tmp_path)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"engineering_fixture": True})
    monkeypatch.setattr(parallel, "run_system", fake_worker)
    config = Config(candidate_count=8, budget=4, initial_count=2, scenarios=("car_following",))
    systems = CURRENT_INDEPENDENT_SUTS[:2]
    root = tmp_path / "parallel"
    rows = parallel.run(
        root=root, config=config, systems=systems, workers=2, dataset_manifest=manifest
    )
    assert len(rows) == 20
    assert [r["sut_id"] for r in rows] == [systems[0].sut_id] * 10 + [systems[1].sut_id] * 10
    assert [r["method"] for r in rows[:10]] == [*pilot.MAIN_METHODS, *pilot.CONTROLS]
    assert all((root / row["result_path"]).is_file() for row in rows)
    assert {row["reused_from"] for row in rows} == {"reused"}
    protocol = read_json(root / "reports/protocol.json")
    assert protocol["execution_layout"]["workers"] == 2
    assert read_json(root / "reports/summary.json")["status"] == "complete"
    batch = root / "tasks" / systems[0].sut_id / "reports/summary.json"
    summary = read_json(batch)
    summary["runs"].pop()
    write_json(batch, summary)
    with pytest.raises(ValueError, match="missing registered tasks"):
        parallel.collect_rows(root, protocol)
    summary["runs"].append(copy.deepcopy(summary["runs"][0]))
    write_json(batch, summary)
    with pytest.raises(ValueError, match="duplicate ADS task"):
        parallel.collect_rows(root, protocol)


def test_parallel_rejects_sequential_and_duplicate_systems(tmp_path):
    with pytest.raises(ValueError, match="fixed-history"):
        parallel.run("sequential", root=tmp_path)
    with pytest.raises(ValueError, match="distinct ADS"):
        parallel.run(root=tmp_path, systems=(CURRENT_INDEPENDENT_SUTS[0],) * 2)


def test_all_selectors_match_serial_and_parallel_and_resume_without_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "OUTPUT", tmp_path)
    manifest = data_manifest(tmp_path)
    for name, options in REDUCED.items():
        module = pilot.BASELINES[name]
        monkeypatch.setattr(module, "DEFAULT", replace(module.DEFAULT, **options))
    monkeypatch.setattr(parallel, "run_system", engineering_worker)
    config = Config(candidate_count=8, budget=5, initial_count=3, scenarios=("car_following",))
    options = dict(config=config, systems=CURRENT_INDEPENDENT_SUTS[:2], dataset_manifest=manifest)
    serial_root, parallel_root = tmp_path / "serial", tmp_path / "parallel"
    serial = parallel.run(root=serial_root, workers=1, **options)
    concurrent = parallel.run(root=parallel_root, workers=2, **options)
    assert len(serial) == len(concurrent) == 20
    timed = {
        "equivalent_execution_s",
        "selection_s",
        "adaptation_s",
        "rss_compute_s",
        "replay_wall_s",
        "cumulative_costs_s",
        "equivalent_testing_curve_s",
    }
    for before, after in zip(serial, concurrent, strict=True):
        first = read_json(serial_root / before["result_path"])
        second = read_json(parallel_root / after["result_path"])
        assert first["specification"] == second["specification"]
        assert first["result"]["selected_indices"] == second["result"]["selected_indices"]
        assert first["evaluation"] == second["evaluation"]
        assert {k: v for k, v in first["metrics"].items() if k not in timed} == {
            k: v for k, v in second["metrics"].items() if k not in timed
        }
        assert first["result"].get("prediction_snapshots") == second["result"].get(
            "prediction_snapshots"
        )
        for a, b in zip(first["result"]["queries"], second["result"]["queries"], strict=True):
            assert a.get("prequery_probability") == b.get("prequery_probability")
            assert a.get("component_weights_after_query") == b.get("component_weights_after_query")
    paths = [parallel_root / row["result_path"] for row in concurrent]
    times = {path: path.stat().st_mtime_ns for path in paths}
    assert parallel.run(root=parallel_root, workers=2, **options) == concurrent
    assert times == {path: path.stat().st_mtime_ns for path in paths}


def test_nested_scale_aggregation_keeps_size_then_ads_order(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "OUTPUT", tmp_path)
    monkeypatch.setattr(parallel, "run_system", fake_worker)
    monkeypatch.setattr(parallel.candidate_sensitivity, "SIZES", (4, 8))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"engineering_fixture": True})
    config = Config(budget=3, initial_count=1, scenarios=("front_brake",))
    root = tmp_path / "sensitivity"
    rows = parallel.run(
        "sensitivity",
        root,
        config=config,
        systems=CURRENT_INDEPENDENT_SUTS[:2],
        workers=2,
        dataset_manifest=manifest,
    )
    assert len(rows) == 32
    assert [row["candidate_count"] for row in rows] == [4] * 16 + [8] * 16
    summary = read_json(root / "reports/summary.json")
    assert summary["status"] == "complete"
    assert read_json(root / "reports/protocol.json")["initial_pool_size"] == 4
    assert summary["physical_truth_records"] == 16 and summary["logical_queries"] == 96
    for size in (4, 8):
        view = root / str(size)
        view_summary = read_json(view / "reports/summary.json")
        assert view_summary["status"] == "complete" and len(view_summary["runs"]) == 16
        assert all((view / row["result_path"]).is_file() for row in view_summary["runs"])
        assert {row["candidate_count"] for row in view_summary["runs"]} == {size}
        view_protocol = read_json(view / "reports/protocol.json")
        assert view_protocol["config"]["candidate_count"] == size
        assert view_protocol["initial_pool_size"] == 4
        assert "candidate_sizes" not in view_protocol
    nested = root / "tasks" / CURRENT_INDEPENDENT_SUTS[0].sut_id / "4/reports/protocol.json"
    changed = read_json(nested)
    changed["config"]["budget"] += 1
    write_json(nested, changed)
    with pytest.raises(ValueError, match="Nested comparison changed"):
        parallel.collect_rows(root, read_json(root / "reports/protocol.json"))


def test_mechanism_models_prepare_before_independent_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "OUTPUT", tmp_path)
    monkeypatch.setattr(parallel, "run_system", fake_worker)
    prepared = []
    monkeypatch.setattr(parallel.mechanisms, "historical_collision_models", prepared.append)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"engineering_fixture": True})
    config = Config(candidate_count=8, budget=3, initial_count=1, scenarios=("car_following",))
    root = tmp_path / "mechanisms"
    rows = parallel.run(
        "mechanisms",
        root,
        config=config,
        systems=CURRENT_INDEPENDENT_SUTS[:2],
        workers=2,
        dataset_manifest=manifest,
    )
    assert prepared == ["car_following"]
    assert len(rows) == 10
    assert {row["method"] for row in rows} == {"joint", *pilot.MECHANISMS}
