import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from controllers.highway.control import Control
from experiments import pilot
from experiments.joint_validation import Config
from experiments.rss import measurement_contract
from experiments.scenarios import candidate_pool
from experiments.storage import read_json, save_archive, write_json
from simulator.highway.collision_env import execute
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer.model import JointResponseKernel


def histories(kind, count=12):
    scenes = candidate_pool(kind, count, replicate=98)
    return [
        ResponseArchive(
            f"history_{i}",
            "rule",
            f"route_{i}",
            "history",
            scenes,
            np.arange(count) % 3 == i,
            np.ones((count, 2)),
            rss_margin=np.linspace(-0.8, 0.4, count),
            rss_contract=measurement_contract(),
        )
        for i in range(2)
    ]


class Brake:
    def act(self, world):
        return Control(-8, 0)


class Bank:
    def __init__(self, candidates, root=None):
        self.root = root
        self.candidates = candidates
        self.manifest = {"contract": {"engineering_fixture": True}, "environment_seed": 81001}
        self.executions = {s.scenario_id: execute(Brake(), s, duration_s=2) for s in candidates}

    def __call__(self, scene):
        return self.executions[scene.scenario_id]

    def execution_path(self, index):
        path = self.root / f"{index:06d}.json.gz"
        scene = self.candidates[index]
        write_json(path, {"scenario": asdict(scene), "trajectory": self(scene).trajectory})
        return path


@pytest.mark.parametrize("names", (("indago", "scenario_fuzz"), ("scenario_fuzz",)))
def test_history_classifiers_load_existing_models_without_retraining_or_rewriting(
    tmp_path, monkeypatch, names
):
    import torch

    monkeypatch.setattr(
        pilot.indago_model, "DEFAULT", replace(pilot.indago_model.DEFAULT, epochs=1)
    )
    monkeypatch.setattr(
        pilot.fuzz_model,
        "DEFAULT",
        replace(pilot.fuzz_model.DEFAULT, epochs=1, hidden_dim=4, heads=1),
    )
    history = histories("car_following", 8)
    original = pilot.historical_classifiers(history, tmp_path, names=names)
    assert set(original) == set(names)
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == set(names)
    paths = list(tmp_path.rglob("*.*"))
    times = {path: path.stat().st_mtime_ns for path in paths}

    def reject_training(*args, **kwargs):
        pytest.fail("Shared classifiers must be loaded without new training")

    for module in (pilot.indago_model, pilot.fuzz_model):
        monkeypatch.setattr(module, "train", reject_training)
    loaded = pilot.historical_classifiers(history, tmp_path, train_missing=False, names=names)
    for name in original:
        for key, tensor in original[name].state_dict().items():
            assert torch.equal(tensor, loaded[name].state_dict()[key])
    assert times == {path: path.stat().st_mtime_ns for path in paths}
    with pytest.raises(FileNotFoundError):
        pilot.historical_classifiers(
            history, tmp_path / "missing", train_missing=False, names=names
        )
    changed = copy.deepcopy(history)
    changed[0].rss_margin[0] += 0.1
    with pytest.raises(ValueError, match="historical construction"):
        pilot.historical_classifiers(changed, tmp_path, train_missing=False, names=names)


@pytest.mark.parametrize("method", tuple(pilot.BASELINES))
@pytest.mark.parametrize("initial_pool_size", (None, 7))
def test_all_seven_baselines_use_the_common_paid_evaluator_and_resume(
    tmp_path, monkeypatch, method, initial_pool_size
):
    import torch

    from testing_methods.baselines.scenario_fuzz.graphs import graph

    torch.set_num_threads(1)
    scenes = candidate_pool("car_following", 12, replicate=99)
    bank = Bank(scenes)
    config = Config(candidate_count=12, initial_count=3, budget=5, scenarios=("car_following",))
    reduced = {
        "av_bo_rf_ei": dict(jackknife_folds=2, tuning_trials=1, n_estimators=4),
        "samota": dict(rbf_epochs=2),
        "scenario_fuzz": dict(cycles=1, mutants=4, executions=1, filter_attempts=1),
        "paired_system_active_evaluation": dict(
            training_epochs=2, mc_draws=4, shortlist_size=4, local_samples=8, paired_samples=2
        ),
        "indago": dict(neighborhood_size=3, max_neighborhoods=2, max_stagnation=2),
    }
    if method in reduced:
        module = pilot.BASELINES[method]
        monkeypatch.setattr(module, "DEFAULT", replace(module.DEFAULT, **reduced[method]))
    classifiers = {}
    if method == "indago":
        classifiers[method] = pilot.indago_model.FailurePredictor(3).eval()
    elif method == "scenario_fuzz":
        model = pilot.fuzz_model.ScenarioRiskNetwork(hidden_dim=4, heads=1).eval()
        model.fit_normalization([graph(s) for s in scenes])
        classifiers[method] = model
    task = {"method": method, "scenario": "car_following", "seed": 11}
    saved, ledger = pilot.baseline_task(
        tmp_path,
        bank,
        histories("car_following"),
        classifiers,
        config,
        11,
        task,
        initial_pool_size=initial_pool_size,
    )
    assert len(ledger.records) == 5
    assert len(set(saved["result"]["selected_indices"])) == 5
    assert saved["result"]["selected_indices"][:3] == pilot.initial_indices(
        12, 3, 11, pool_size=initial_pool_size
    )
    if initial_pool_size is not None:
        assert saved["specification"]["initial_indices"] == saved["result"]["selected_indices"][:3]
    assert len(saved["metrics"]["discovery_curve"]) == 5
    assert saved["evaluation"]["candidate_collisions"] == [
        bank(scene).feedback.collision for scene in scenes
    ]
    for record in ledger.records:
        feedback = record["feedback"]
        assert feedback["safety"]["rss_margin"] is not None
        assert feedback["trajectory"]
        if feedback["collision"]:
            assert feedback["safety"]["impact_delta_v_mps"] is not None
        else:
            assert feedback["safety"]["driving_quality_score"] is not None
    resumed, again = pilot.baseline_task(
        tmp_path,
        bank,
        histories("car_following"),
        classifiers,
        config,
        11,
        task,
        initial_pool_size=initial_pool_size,
    )
    assert resumed == json.loads(json.dumps(saved))
    assert len(again.records) == 5


@pytest.mark.parametrize(
    "systems,suite", [(pilot.SYSTEMS, "pilot"), (CURRENT_INDEPENDENT_SUTS, "fixed")]
)
def test_fixed_comparison_registers_all_methods_and_seals_completed_tasks(
    tmp_path,
    monkeypatch,
    systems,
    suite,
):
    import torch

    entries = []
    for kind in pilot.DEFAULT.scenarios:
        for archive in histories(kind, 4):
            path = tmp_path / "data" / kind / f"{archive.source_id}.npz"
            save_archive(path, archive, provenance=archive.provenance)
            entries.append(
                {
                    "scenario": kind,
                    "source_id": archive.source_id,
                    "path": str(path.relative_to(tmp_path / "data")),
                    "fingerprint": archive_fingerprints([archive])[archive.source_id],
                }
            )
    manifest = tmp_path / "data/manifest.json"
    write_json(manifest, {"archives": entries})
    calls, archives, prepared, banks = [], [], [], []

    def fake_main(destination, bank, models, history, config, seed, **kwargs):
        task = kwargs["snapshot"]
        calls.append((task, list(history), models, kwargs["native_feedback"]))
        return {"metrics": {}, "result": {"resources": {}}, "specification": {}}, None

    def fake_baseline(destination, bank, history, classifiers, config, seed, task, **kwargs):
        calls.append((task, list(history), None, True))
        return {"metrics": {}, "result": {"resources": {}}, "specification": {}}, None

    def fake_bank(root, sut, kind, config, **kwargs):
        banks.append((sut.sut_id, kind))
        return SimpleNamespace(manifest={"contract": {"fixture": True}})

    monkeypatch.setattr(pilot, "run_one", fake_main)
    monkeypatch.setattr(pilot, "baseline_task", fake_baseline)
    monkeypatch.setattr(pilot, "target_bank", fake_bank)
    monkeypatch.setattr(pilot, "finalize_task", lambda *args, **kwargs: archives.append(kwargs))

    def builder(history, destination):
        assert torch.get_num_threads() == 1
        prepared.append((history[0].scenarios[0].kind, destination))
        return {}

    root = tmp_path / "pilot"
    torch.set_num_threads(2)
    result = pilot.fixed_comparison(
        root,
        dataset_manifest=manifest,
        model_loader=lambda kind: (kind, kind),
        classifier_builder=builder,
        systems=systems,
        suite=suite,
    )
    tasks = len(systems) * 4
    assert len(result) == len(calls) == len(archives) == tasks * 10
    assert len(banks) == tasks and len(prepared) == 4
    assert sum(r["comparison"] == "main" for r in result) == tasks * 8
    assert sum(r["comparison"] == "control" for r in result) == tasks * 2
    for task, history, models, native in calls:
        assert task["seed"] == 11 and task["held_family"] is None and native
        if task["method"] == "no_history":
            assert history == [] and models == (None, None)
        else:
            assert len(history) == 2
            assert all(
                a.role == "history" and a.scenarios[0].kind == task["scenario"] for a in history
            )
    protocol = read_json(root / "reports/protocol.json")
    assert protocol["config"]["budget"] == 200 and protocol["config"]["initial_count"] == 20
    assert protocol["candidate_replicate"] == 10
    assert protocol["cpu_torch_threads"] == 1
    assert protocol["main_runs"] == tasks * 8 and protocol["control_runs"] == tasks * 2
    assert protocol["systems"] == [asdict(s) for s in systems]
    assert read_json(root / "reports/summary.json")["status"] == "complete"


def test_full_comparison_reuses_verified_pilot_runs_without_additional_queries(
    tmp_path, monkeypatch
):
    import torch

    from experiments import joint_validation

    torch.set_num_threads(1)
    torch.manual_seed(1301)
    entries = []
    for archive in histories("car_following", 5):
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
    config = Config(candidate_count=5, initial_count=1, budget=2, scenarios=("car_following",))
    bank = Bank(candidate_pool("car_following", 5), tmp_path / "physical")
    monkeypatch.setattr(pilot, "target_bank", lambda *args, **kwargs: bank)
    models = tuple(JointResponseKernel(3).double() for _ in range(2))
    calls = []
    isolated = joint_validation.run_isolated

    def observed(*args, **kwargs):
        calls.append(True)
        return isolated(*args, **kwargs)

    monkeypatch.setattr(joint_validation, "run_isolated", observed)
    options = dict(
        config=config,
        dataset_manifest=manifest,
        model_loader=lambda kind: copy.deepcopy(models),
        methods=("joint", "random", "collision", "no_history"),
    )
    source = tmp_path / "pilot"
    original = pilot.fixed_comparison(source, systems=pilot.SYSTEMS[:1], **options)
    assert len(calls) == 4
    full = tmp_path / "fixed"
    rows = pilot.fixed_comparison(
        full, systems=pilot.SYSTEMS, suite="fixed", reuse_root=source, **options
    )
    assert len(rows) == 8 and len(calls) == 8
    assert [r["metrics"] for r in rows[:4]] == [r["metrics"] for r in original]
    assert all(r["reused_from"] == "../pilot" for r in rows[:4])
    assert all(r["reused_from"] is None for r in rows[4:])
    assert all((full / r["result_path"]).is_file() for r in rows)
    with pytest.raises(ValueError, match="differs in config"):
        pilot.fixed_comparison(
            tmp_path / "changed",
            systems=pilot.SYSTEMS,
            reuse_root=source,
            **{**options, "config": replace(config, budget=3)},
        )
