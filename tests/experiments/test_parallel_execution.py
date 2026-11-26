"""Parallel routing and archive writes without running physical simulations."""

import gzip
import json
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from controllers.highway.catalog import BY_ID
from controllers.highway.factory import create
from controllers.highway.learning import CHECKPOINT_ROOT
from experiments import executions, history_transfer_experiment, parallel
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from simulator.highway.observation import VehicleState, VisibleWorld


def _thread_probe():
    from threadpoolctl import threadpool_info

    return {
        "pid": os.getpid(),
        "torch": torch.get_num_threads(),
        "blas": [p["num_threads"] for p in threadpool_info()],
        "inherited": os.environ["OPENBLAS_NUM_THREADS"],
    }


def test_spawn_workers_apply_inherited_and_loaded_library_limits():
    parallel.prepare_processes()
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=mp.get_context("spawn"),
        initializer=parallel.initialize_worker,
    ) as pool:
        futures = [pool.submit(_thread_probe) for _ in range(4)]
        rows = [future.result() for future in futures]
    assert all(r["pid"] != os.getpid() for r in rows)
    assert all(r["torch"] == 1 and r["inherited"] == "1" for r in rows)
    assert all(n == 1 for r in rows for n in r["blas"])


@pytest.mark.parametrize(
    "sut_id", ["history_ppo", "history_dqn_native", "target_ppo", "target_dqn"]
)
def test_learning_weights_reused_with_fresh_driver_and_identical_actions(sut_id, monkeypatch):
    if not (CHECKPOINT_ROOT / BY_ID[sut_id].training_run / "best.zip").exists():
        pytest.skip("Registered ADS checkpoint is not installed")
    monkeypatch.setattr(executions, "_LEARNING_SUT", sut_id)
    monkeypatch.setattr(executions, "_LEARNING_TEMPLATE", None)
    loads = []

    def counted_create(name):
        loads.append(name)
        return create(name)

    monkeypatch.setattr(executions, "create", counted_create)
    cached = executions._execution_policy(sut_id)
    fresh = create(sut_id)
    before = {k: v.clone() for k, v in cached.policy.model.policy.state_dict().items()}
    for index in range(12):
        world = VisibleWorld(
            index / 5,
            VehicleState("ego", 100 + index * 4, 4, 20, 0),
            (VehicleState("lead", 130 + index * 3, 4, 15, 0),),
            (0.0, 4.0),
        )
        assert cached.act(world) == fresh.act(world)
    another = executions._execution_policy(sut_id)
    assert another.policy.model is cached.policy.model
    assert another.policy.driver is not cached.policy.driver
    assert another.policy.driver.target_lane is None
    assert another.policy.driver.target_speed is None
    assert executions._LEARNING_TEMPLATE.policy.driver.target_lane is None
    assert loads == [sut_id]
    assert all(
        torch.equal(v, cached.policy.model.policy.state_dict()[k]) for k, v in before.items()
    )


def test_non_learning_controllers_are_created_fresh(monkeypatch):
    monkeypatch.setattr(executions, "_LEARNING_SUT", None)
    first = executions._execution_policy("history_idm_mobil")
    second = executions._execution_policy("history_idm_mobil")
    assert first is not second and first.policy is not second.policy


def test_worker_writes_the_full_atomic_archive_and_returns_only_timing(tmp_path, monkeypatch):
    scene = Scenario("unit", "car_following", {"ego_speed_mps": 20})
    trajectory = [{"unit": True}]
    feedback = {"collision": False, "execution_s": 0.25}
    monkeypatch.setattr(
        executions,
        "_execute_task",
        lambda *args: SimpleNamespace(
            feedback=SimpleNamespace(execution_s=0.25, to_dict=lambda: feedback),
            trajectory=trajectory,
        ),
    )
    monkeypatch.setattr(executions, "analyze_events", lambda *args: {"unit_event": 1})
    target = tmp_path / "execution.json.gz"
    elapsed = executions._collect_task("unit", scene, 11, tmp_path / "attempt.json", 7, target)
    assert elapsed == 0.25
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        assert json.load(stream) == {
            "call": 7,
            "scenario": asdict(scene),
            "feedback": feedback,
            "trajectory": trajectory,
            "events": {"unit_event": 1},
        }
    assert not target.with_suffix(".tmp").exists()
    assert not (tmp_path / "calls.jsonl").exists()


def test_failed_worker_does_not_publish_an_execution(tmp_path, monkeypatch):
    def failed(*args):
        raise ValueError("unit execution failed")

    monkeypatch.setattr(executions, "_execute_task", failed)
    target = tmp_path / "execution.json.gz"
    with pytest.raises(ValueError, match="unit execution failed"):
        executions._collect_task("unit", None, 11, tmp_path / "attempt.json", 7, target)
    assert not target.exists()


def test_parallel_collection_keeps_one_ledger_and_resumes_without_new_calls(tmp_path, monkeypatch):
    scenes = [Scenario(f"unit-{i}", "car_following", {"ego_speed_mps": 20 + i}) for i in range(3)]
    monkeypatch.setattr(executions, "require_qualified", lambda sut: "unit qualification")
    monkeypatch.setattr(executions, "execution_contract", lambda *a, **kw: {"unit": True})
    monkeypatch.setattr(executions, "create", lambda sut: object())
    monkeypatch.setattr(executions, "analyze_events", lambda *a: {"unit": True})

    def measured_fixture(policy, scene, **kwargs):
        return SimpleNamespace(
            feedback=QueryFeedback(scene.scenario_id, False, SafetyMeasurements(10, 2), 0),
            trajectory=[{"unit": True}],
        )

    monkeypatch.setattr(executions, "execute", measured_fixture)

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["initializer"] is executions._initialize_collection
            assert kwargs["initargs"] == ("history_idm_mobil",)
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            assert fn is executions._collect_task
            future = Future()
            future.set_result(fn(*args))
            return future

    monkeypatch.setattr(executions, "ProcessPoolExecutor", Pool)
    bank = executions.collect_bank("history_idm_mobil", scenes, tmp_path, workers=2)
    assert bank.manifest["attempted_calls"] == bank.manifest["successful_calls"] == 3
    assert [bank(scene).feedback.scenario_id for scene in scenes] == [s.scenario_id for s in scenes]
    ledger = (tmp_path / "calls.jsonl").read_bytes()
    records = [json.loads(line) for line in ledger.splitlines()]
    assert sorted(r["call"] for r in records if r["status"] == "complete") == [1, 2, 3]
    monkeypatch.setattr(executions, "ProcessPoolExecutor", lambda **kw: pytest.fail("resubmitted"))
    resumed = executions.collect_bank("history_idm_mobil", scenes, tmp_path, workers=2)
    assert resumed.manifest["attempted_calls"] == 3
    assert (tmp_path / "calls.jsonl").read_bytes() == ledger


def test_training_preflights_all_scenes_before_parallel_dispatch(tmp_path, monkeypatch):
    module, entry, job = history_transfer_experiment, "train_models", "train_scenario"
    checked, submitted = [], []
    scenes = ("car_following", "cut_out")
    monkeypatch.setattr(module, "scenarios_for", lambda root: scenes)
    monkeypatch.setattr(module, "register", lambda root: None)
    monkeypatch.setattr(module, "scenario_history", lambda root, kind: checked.append(kind))
    monkeypatch.setattr(module, job, lambda kind, root: {"scenario": kind})

    class Pool:
        def __init__(self, **kwargs):
            assert checked == list(scenes)
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["initializer"] is parallel.initialize_worker

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, kind, root):
            submitted.append(kind)
            future = Future()
            future.set_result(fn(kind, root))
            return future

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    getattr(module, entry)(tmp_path)
    assert submitted == list(scenes)


def test_fixed_protocol_is_not_rewritten_by_repeated_readers(tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"
    value = {"seed": 11}
    history_transfer_experiment.freeze(path, value)
    monkeypatch.setattr(
        history_transfer_experiment, "write_json", lambda *a: pytest.fail("rewrite")
    )
    assert history_transfer_experiment.freeze(path, value) == value
    with pytest.raises(ValueError, match="Registered protocol changed"):
        history_transfer_experiment.freeze(path, {"seed": 23})


def test_gpu_jobs_are_serial_and_unavailable_cuda_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert parallel.training_workers("cuda") == 1
    assert parallel.training_workers("cpu") == parallel.TRAINING_WORKERS
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        parallel.training_workers("cuda")
