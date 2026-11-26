"""Device declarations and separate pools without running native simulations."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments import history_transfer_experiment as study
from experiments import parallel, transfer_batches, transfer_execution
from experiments.k_shot import split_candidates
from experiments.scenarios import sample
from experiments.transfer_settings import native_settings
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_pooled_has_one_cuda_worker_while_small_adaptation_stays_parallel(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert parallel.prediction_device("pooled_history") == "cuda"
    assert parallel.prediction_workers("pooled_history", 8) == 1
    for method in ("target_only", "full"):
        assert parallel.prediction_device(method) == "cpu"
        assert parallel.prediction_workers(method, 8) == 8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        parallel.prediction_workers("pooled_history")
    assert parallel.prediction_workers("full", 4) == 4


@pytest.mark.parametrize("method", ["target_only", "pooled_history", "full"])
def test_registered_device_reaches_isolated_prediction_selector(method, monkeypatch, tmp_path):
    bank = SimpleNamespace(
        candidates=sample("car_following", 8, "new", replicate=92),
        manifest={"sut_id": "unit", "contract": {}},
        snapshot_fingerprint="unit",
    )
    split = split_candidates(bank.candidates, 11, budget=3)
    models = (None, None) if method == "target_only" else (SharedPrivateKernel(3).double(), None)
    history = {} if method == "pooled_history" else None
    sources = {"specification": {}} if method == "full" else None
    settings = native_settings()
    case = transfer_execution.prediction_case(
        bank,
        split,
        11,
        method,
        models,
        history,
        sources,
        settings=settings,
        checkpoints=(0, 3),
        limits={"wall_s": 120, "child_memory_bytes": 2**30},
        evaluate=False,
    )
    captured = {}

    def capture(root, declared, bank, selector, public, **kwargs):
        captured.update(public)
        return public

    monkeypatch.setattr(transfer_execution, "_execute", capture)
    transfer_execution.run_prediction(
        tmp_path,
        case,
        bank,
        models,
        history,
        sources,
        settings=settings,
        paid_execution=lambda i: pytest.fail("No queries in a dispatch test"),
    )
    assert captured["device"] == case.specification["device"] == parallel.prediction_device(method)
    if method == "pooled_history":
        stale = replace(case, specification={**case.specification, "device": "cpu"})
        with pytest.raises(ValueError, match="pre-query task declaration"):
            transfer_execution.run_prediction(
                tmp_path, stale, bank, models, history, sources,
                settings=settings, paid_execution=None,
            )


@pytest.mark.parametrize("mode", ["prediction", "budget"])
def test_formal_batch_uses_separate_cpu_and_single_cuda_pools(mode, monkeypatch, tmp_path):
    methods = ["full", "target_only"] + (["pooled_history"] if mode == "prediction" else ["random"])
    cases = [SimpleNamespace(method=m, key=(m,)) for m in methods for _ in range(2)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(study, "require_method_evaluation", lambda root: None)
    monkeypatch.setattr(transfer_batches, "load_settings", lambda root: None)
    monkeypatch.setattr(transfer_batches, "limits_from_acceptance", lambda root: None)
    monkeypatch.setattr(transfer_batches, "scenarios_for", lambda root: ("car_following",))
    monkeypatch.setattr(transfer_batches, "prepare_context", lambda *a: ({}, {}))
    monkeypatch.setattr(transfer_batches, "make_cases", lambda *a: cases)
    monkeypatch.setattr(transfer_batches, "declare_batch", lambda *a: (tmp_path, {}))
    pools, progress = [], []
    monkeypatch.setattr(transfer_batches, "write_json", lambda path, value: progress.append(value))

    class Pool:
        def __init__(self, **kwargs):
            self.workers, self.methods = kwargs["max_workers"], []
            assert kwargs["mp_context"].get_start_method() == "spawn"
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, case):
            self.methods.append(case.method)
            result = Future()
            result.set_result({"key": list(case.key), "status": "complete"})
            return result

    monkeypatch.setattr(transfer_batches, "ProcessPoolExecutor", Pool)
    if mode == "budget":
        with pytest.raises(ValueError, match="deferred"):
            transfer_batches.run(mode, tmp_path)
        assert pools == [] and progress == []
        return
    transfer_batches.run(mode, tmp_path)
    assert pools[0].workers == parallel.METHOD_WORKERS
    assert "pooled_history" not in pools[0].methods
    if mode == "prediction":
        assert len(pools) == 2
        assert pools[1].workers == 1 and pools[1].methods == ["pooled_history"] * 2
    else:
        assert len(pools) == 1
    assert len(progress[-1]["completed"]) == len(cases)


def test_prediction_output_cannot_claim_another_registered_device():
    case = SimpleNamespace(experiment="prediction", specification={"device": "cuda"})
    result = {"selected_indices": [], "queries": [], "resources": {"device": "cpu"}}
    with pytest.raises(ValueError, match="Prediction device"):
        transfer_execution._validate_output(case, result, [])


def test_cost_rows_keep_gpu_memory_separate():
    from experiments.collect_transfer_records import _cost_row

    case = SimpleNamespace(
        experiment="prediction",
        sut_id="unit",
        scenario="car_following",
        seed=11,
        method="pooled_history",
    )
    row = _cost_row(
        case,
        {
            "query_ledger": [],
            "result": {
                "resources": {
                    "device": "cuda",
                    "sampled_peak_cpu_bytes": 1024,
                    "peak_cuda_allocated_bytes": 4096,
                    "wall_s": 0.2,
                }
            },
        },
    )
    assert row["selector_device"] == "cuda"
    assert row["selector_peak_cpu_bytes"] == 1024
    assert row["selector_peak_cuda_allocated_bytes"] == 4096
