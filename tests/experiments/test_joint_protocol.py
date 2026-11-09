import copy
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from controllers.highway.catalog import INDEPENDENT_SUTS
from controllers.highway.control import Control
from experiments.history import load_training_archives, model_version, update_shared_model
from experiments.joint_validation import Config, run_one, sequential, update_scenario
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from experiments.storage import read_json, save_archive, write_json
from simulator.highway.collision_env import execute
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.training import Config as TrainingConfig


def archive(name, role, offset=0, *, kind="car_following"):
    scenes = sample(kind, 8, role, replicate=offset)
    labels = (np.arange(8) + offset) % 3 == 0
    return ResponseArchive(
        name,
        "rule",
        name,
        role,
        scenes,
        labels,
        np.ones((8, 2)),
        rss_margin=np.where(labels, -0.8, 0.2),
        rss_contract=measurement_contract(),
    )


def test_shared_model_publication_uses_final_steps_without_development_and_preserves_parent(
    tmp_path,
):
    torch.set_num_threads(1)
    torch.manual_seed(1301)
    models = tuple(JointResponseKernel(3).double() for _ in range(2))
    versions = [model_version(m) for m in models]
    history = [archive("old", "history"), archive("new", "history", 1)]
    config = TrainingConfig(
        steps=1, initial_count=1, budget=3, historical_context=2, diagnostic_interval=1
    )
    active, result = update_shared_model(
        tmp_path / "candidate",
        history,
        models,
        new_source_ids=["new"],
        config=config,
    )
    assert result["accepted"]
    assert [model_version(m) for m in models] == versions
    assert (tmp_path / "active.json").exists()
    assert [model_version(m) for m in active] != versions
    assert result["reason"] == "finite_final_fixed_step_model"


def test_parallel_scenario_updates_match_serial_models_and_paid_task_replay(tmp_path):
    torch.set_num_threads(1)
    training = TrainingConfig(
        steps=1, initial_count=1, budget=3, historical_context=2, diagnostic_interval=1
    )
    jobs, versions = [], []
    for index, kind in enumerate(("car_following", "front_brake")):
        archives = []
        for offset, name in enumerate(("old_a", "old_b", "paid")):
            role = "target" if name == "paid" else "history"
            scenes = sample(kind, 4, "new" if role == "target" else role, replicate=offset)
            labels = (np.arange(4) + offset) % 3 == 0
            archives.append(
                ResponseArchive(
                    f"{kind}:{name}",
                    "rule",
                    name,
                    role,
                    scenes,
                    labels,
                    np.ones((4, 2)),
                    rss_margin=np.where(labels, -0.8, 0.2),
                    rss_contract=measurement_contract(),
                    provenance={
                        "completed_task": role == "target",
                        "record_ids": [f"{kind}:{name}:{i}" for i in range(4)],
                    },
                )
            )
        torch.manual_seed(900 + index)
        models = tuple(JointResponseKernel(archives[0].x.shape[1]).double() for _ in range(2))
        versions.append([model_version(model) for model in models])
        jobs.append(
            {
                "destination": tmp_path / "parallel" / kind,
                "history": archives,
                "models": models,
                "completed": archives[-1:],
                "new_source_ids": [archives[-1].source_id],
                "config": Config(
                    candidate_count=4, budget=4, initial_count=1, scenarios=(kind,), batch_size=2
                ),
                "training_config": training,
            }
        )
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update_scenario, **job) for job in jobs]
        parallel = [future.result() for future in futures]
    for job, original, (models, publication) in zip(jobs, versions, parallel, strict=True):
        kind = job["config"].scenarios[0]
        serial_root = tmp_path / "serial" / kind
        reference, serial_publication = update_scenario(**{**job, "destination": serial_root})
        assert publication["accepted"] and serial_publication["accepted"]
        assert publication["specification"] == serial_publication["specification"]
        assert publication["specification"]["scenario"] == kind
        assert set(publication["specification"]["history_fingerprints"]) == {
            archive.source_id for archive in job["history"]
        }
        assert [model_version(model) for model in job["models"]] == original
        assert [model_version(model) for model in models] == [
            model_version(model) for model in reference
        ]
        actual = read_json(job["destination"] / "retention.json")
        expected = read_json(serial_root / "retention.json")
        assert actual["new_physical_calls"] == actual["new_paid_queries"] == 0
        assert actual["specification"] == expected["specification"]
        for left, right in zip(actual["tasks"], expected["tasks"], strict=True):
            assert left["record_ids"] == right["record_ids"]
            assert left["after"]["shared_training_contains_task"]
            for moment in ("before", "after"):
                np.testing.assert_array_equal(
                    left[moment]["prequery_probabilities"], right[moment]["prequery_probabilities"]
                )


class Brake:
    def act(self, world):
        return Control(-8, 0)


class Bank:
    def __init__(self, candidates, root=None):
        self.root = root
        self.candidates = candidates
        self.manifest = {
            "contract": {"fixture": "short real trajectories"},
            "environment_seed": 81001,
        }
        self.executions = {s.scenario_id: execute(Brake(), s, duration_s=0.4) for s in candidates}

    def __call__(self, scene):
        return self.executions[scene.scenario_id]

    def execution_path(self, index):
        path = self.root / f"{index:06d}.json.gz"
        scene = self.candidates[index]
        write_json(path, {"scenario": asdict(scene), "trajectory": self(scene).trajectory})
        return path


def test_no_history_resume_rejects_changed_candidate_coordinates_and_environment(tmp_path):
    scenes = sample("car_following", 4, "new", replicate=20)
    bank = Bank(scenes)
    models = (None, None)
    config = Config(candidate_count=4, budget=2, initial_count=1)
    saved, _ = run_one(tmp_path, bank, models, [], config, 11)
    assert saved["specification"]["target_model_version"] is None
    assert saved["specification"]["encoder_version"] is None
    assert saved["result"]["component_names"] == ["raw_target"]
    resumed, _ = run_one(tmp_path, bank, models, [], config, 11)
    assert resumed == json.loads(json.dumps(saved))
    bank.manifest["environment_seed"] += 1
    with pytest.raises(ValueError, match="snapshot"):
        run_one(tmp_path, bank, models, [], config, 11)
    bank.manifest["environment_seed"] -= 1
    bank.candidates[0] = replace(scenes[0], parameters={**scenes[0].parameters, "gap_m": 60})
    with pytest.raises(ValueError, match="snapshot"):
        run_one(tmp_path, bank, models, [], config, 11)


def test_short_candidate_pool_caps_initialization_and_paid_budget(tmp_path):
    scenes = sample("car_following", 4, "new", replicate=20)
    models = tuple(JointResponseKernel(3).double() for _ in range(2))
    saved, ledger = run_one(tmp_path, Bank(scenes), models, [], Config(), 11)
    assert saved["specification"]["budget"] == 4
    assert saved["specification"]["initial_count"] == 4
    assert len(ledger.records) == 4
    assert len(set(saved["result"]["selected_indices"])) == 4


def test_target_tasks_keep_identical_initial_candidates_across_nested_sizes(tmp_path):
    from experiments.scenarios import candidate_pool

    initial_ids = []
    for size in (4, 7, 9):
        scenes = candidate_pool("car_following", size, replicate=30)
        bank = Bank(scenes)
        config = Config(candidate_count=size, budget=3, initial_count=2)
        saved, _ = run_one(
            tmp_path / str(size), bank, (None, None), [], config, 11, initial_pool_size=4
        )
        prefix = saved["result"]["selected_indices"][:2]
        assert saved["specification"]["initial_pool_size"] == 4
        assert saved["specification"]["initial_indices"] == prefix
        initial_ids.append([scenes[i].scenario_id for i in prefix])
        if size > 4:
            with pytest.raises(ValueError, match="snapshot"):
                run_one(
                    tmp_path / str(size),
                    bank,
                    (None, None),
                    [],
                    config,
                    11,
                    initial_pool_size=5,
                )
    assert initial_ids[0] == initial_ids[1] == initial_ids[2]


def test_run_rejects_model_trained_outside_admitted_history(tmp_path):
    scenes = sample("car_following", 4, "new", replicate=20)
    models = tuple(JointResponseKernel(3).double() for _ in range(2))
    models[1].construction_specification = {"history_fingerprints": {"future": "unpaid"}}
    with pytest.raises(ValueError, match="outside the admitted"):
        run_one(
            tmp_path,
            Bank(scenes),
            models,
            [archive("old", "history")],
            Config(candidate_count=4, budget=2, initial_count=1),
            11,
        )


def test_loader_rejects_development_archives(tmp_path):
    development = archive("d", "development")
    save_archive(tmp_path / "dev.npz", development, provenance=development.provenance)
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "archives": [
                {
                    "scenario": "car_following",
                    "source_id": "d",
                    "path": "dev.npz",
                    "fingerprint": archive_fingerprints([development])["d"],
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="identity"):
        load_training_archives(manifest, "car_following")


@pytest.mark.parametrize("kinds", [("car_following",), ("car_following", "front_brake")])
def test_chronological_runner_keeps_regimes_seeds_and_future_systems_separate(
    tmp_path, monkeypatch, kinds
):
    from experiments import joint_validation as workflow

    torch.set_num_threads(1)
    history = [
        archive(f"{kind}_{name}", "history", offset, kind=kind)
        for kind in kinds
        for offset, name in enumerate(("a", "b"))
    ]
    entries = []
    for a in history:
        save_archive(tmp_path / f"{a.source_id}.npz", a, provenance=a.provenance)
        entries.append(
            {
                "source_id": a.source_id,
                "scenario": a.scenarios[0].kind,
                "path": f"{a.source_id}.npz",
                "fingerprint": archive_fingerprints([a])[a.source_id],
            }
        )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"archives": entries})
    banks = {}

    def bank_factory(root, sut, kind, config, **kwargs):
        key = sut.sut_id, kind
        if key not in banks:
            banks[key] = Bank(
                sample(kind, config.candidate_count, "new", replicate=20), root / sut.sut_id / kind
            )
        return banks[key]

    monkeypatch.setattr(workflow, "target_bank", bank_factory)
    dimensions = {a.scenarios[0].kind: a.x.shape[1] for a in history}
    models = {
        kind: tuple(JointResponseKernel(dimensions[kind]).double() for _ in range(2))
        for kind in kinds
    }
    config = Config(
        candidate_count=4,
        budget=2,
        initial_count=1,
        algorithm_seeds=(11,),
        scenarios=kinds,
        update_every_systems=2,
        batch_size=2,
    )
    training = TrainingConfig(
        steps=1, initial_count=1, budget=3, historical_context=2, diagnostic_interval=1
    )
    torch.set_num_threads(2)
    rows, updates = sequential(
        tmp_path / "experiment",
        config=config,
        systems=INDEPENDENT_SUTS[:3],
        dataset_manifest=manifest,
        model_loader=lambda kind: copy.deepcopy(models[kind]),
        training_config=training,
    )
    assert len(rows) == 9 * len(kinds) and len(updates) == len(kinds)
    protocol = read_json(tmp_path / "experiment/reports/protocol.json")
    assert protocol["cpu_torch_threads"] == 1
    assert protocol["shared_update_workers"] == len(kinds)
    assert protocol["shared_update_training"]["steps"] == training.steps
    for update in updates:
        retained = read_json(tmp_path / "experiment" / update["retention_path"])
        assert retained["new_physical_calls"] == retained["new_paid_queries"] == 0
        assert not retained["publication_gate"] and len(retained["tasks"]) == 2
        assert update["publication"]["specification"]["scenario"] == update["scenario"]
        for task in retained["tasks"]:
            assert task["after"]["shared_training_contains_task"]
            assert len(task["before"]["prequery_probabilities"]) == 2
            assert task["after"]["evaluated_records"] == 1
    for regime in ("fixed_history", "append_sources", "update_shared"):
        runs = [r for r in rows if r["regime"] == regime]
        expected = [2, 2, 2] if regime == "fixed_history" else [2, 3, 4]
        assert [len(r["source_ids"]) for r in runs] == [count for count in expected for _ in kinds]
        for r in runs:
            assert not any(source.startswith(r["sut_id"] + "@") for source in r["source_ids"])
    # A restart reads each immutable result/snapshot instead of changing a previous task.
    resumed, _ = sequential(
        tmp_path / "experiment",
        config=config,
        systems=INDEPENDENT_SUTS[:3],
        dataset_manifest=manifest,
        model_loader=lambda kind: copy.deepcopy(models[kind]),
        training_config=training,
    )
    assert [r["metrics"] for r in resumed] == [r["metrics"] for r in rows]
    from experiments.report_joint import report

    summary = report(tmp_path / "experiment")
    assert set(summary["means_by_task"]) == {"fixed_history", "append_sources", "update_shared"}
    assert len(summary["completed_task_replay"]) == 2 * len(kinds)
    assert summary["prediction_diagnostics"]["status"] == "complete"
    assert len(summary["prediction_diagnostics"]["snapshot_scores"]) == len(rows)
    assert any("failure_space_physical.png" in p for p in summary["figures"])
    for kind in kinds:
        assert (
            tmp_path / "experiment/reports/figures/sequential" / f"{kind}_discovery.png"
        ).is_file()
    assert not list((tmp_path / "experiment/reports").rglob("*.pdf"))
