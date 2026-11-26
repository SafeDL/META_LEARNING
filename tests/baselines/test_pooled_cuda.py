"""CUDA keeps the pooled likelihood, ten-step fit and paid-feedback boundaries."""

import copy

import numpy as np
import pytest
import torch

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.transfer_prediction_audit import reconstruct
from experiments.transfer_settings import native_settings
from testing_methods.baselines.response_baselines.pooled_gp import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_pooled_cuda_matches_cpu_and_keeps_base_and_feedback_isolated(backend):
    torch.manual_seed(1301)
    scenes = sample("car_following", 8, "new", replicate=92)
    x = features(scenes)
    base = SharedPrivateKernel(3).double().eval()
    original = copy.deepcopy(base.state_dict())
    history = {
        "x": x[[0, 0, 1, 1]],
        "collision": [False, True, False, False],
        "rss_margin": [0.2, -0.3, 0.4, 0.1],
    }
    settings = native_settings()

    class Oracle:
        def __init__(self, change_future=False):
            self.paid = []
            self.change_future = change_future

        def query(self, index, *, purpose):
            self.paid.append(index)
            collision = index % 2 == 0
            if self.change_future and index == 6:
                collision = not collision
            return QueryFeedback(
                scenes[index].scenario_id,
                collision,
                SafetyMeasurements(1, 1, rss_margin=-0.3 if collision else 0.4),
                0,
                measurement_contract(),
            )

    def run(device, changed=False):
        oracle = Oracle(changed)
        result = select(
            scenes,
            oracle,
            pretrained=base,
            history=history,
            query_order=[3, 1, 6],
            budget=3,
            snapshot_counts=(0, 1, 3),
            device=device,
            inference_backend=backend,
            fit_settings=settings.pooled,
        )
        assert oracle.paid == [3, 1, 6]
        return result

    cpu, gpu = run("cpu"), run("cuda")
    for left, right in zip(cpu["prediction_snapshots"], gpu["prediction_snapshots"]):
        assert left["observed_indices"] == right["observed_indices"]
        assert left["training_count"] == right["training_count"]
        np.testing.assert_allclose(
            left["probabilities"], right["probabilities"], atol=1e-9, rtol=1e-9
        )
        assert left["target_parameters"] == pytest.approx(right["target_parameters"], abs=1e-8)
    assert gpu["prediction_snapshots"][:2] == run("cuda", changed=True)["prediction_snapshots"][:2]
    assert cpu["resources"]["device"] == "cpu"
    assert cpu["resources"]["peak_cuda_allocated_bytes"] == 0
    assert gpu["resources"]["device"] == "cuda"
    assert gpu["resources"]["peak_cuda_allocated_bytes"] > 0
    assert gpu["conditional_inference_device"] == gpu["kernel_and_adaptation_device"] == "cuda"
    assert all(p.device.type == "cpu" for p in base.parameters())
    assert all(torch.equal(original[n], p) for n, p in base.state_dict().items())
    if backend == "grouped":
        records = gpu["queries"]
        margins = [r["feedback"]["safety"]["rss_margin"] for r in records]
        for snapshot in gpu["prediction_snapshots"]:
            actual = reconstruct(
                scenes,
                snapshot,
                records,
                margins,
                method="pooled_history",
                models=(base, None),
                pooled=history,
                sources=None,
                settings=settings,
            )
            np.testing.assert_allclose(actual, snapshot["probabilities"], atol=1e-10, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_pooled_cuda_runs_in_spawn_child_with_only_disclosed_feedback():
    scenes = sample("car_following", 8, "new", replicate=92)
    x = features(scenes)

    def execute(scene):
        return PaidExecution(
            QueryFeedback(
                scene.scenario_id,
                False,
                SafetyMeasurements(1, 1, rss_margin=0.4),
                0,
                measurement_contract(),
            ),
            [],
        )

    ledger = QueryLedger(scenes, 2, execute, allow_registration=False)
    result = run_isolated(
        select,
        {
            "candidates": scenes,
            "pretrained": SharedPrivateKernel(3).double(),
            "history": {"x": x[:2], "collision": [False, True], "rss_margin": [0.2, -0.3]},
            "query_order": [3, 6],
            "budget": 2,
            "snapshot_counts": (0, 2),
            "inference_backend": "grouped",
            "fit_settings": native_settings().pooled,
            "device": "cuda",
        },
        ledger,
        wall_limit_s=60,
        memory_limit_bytes=8 * 1024**3,
    )
    assert result["selected_indices"] == [3, 6]
    assert len(ledger.records) == 2
    assert result["resources"]["device"] == "cuda"
    assert result["resources"]["peak_cuda_allocated_bytes"] > 0
