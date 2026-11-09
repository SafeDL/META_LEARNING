import copy

import numpy as np
import pytest
import torch

from experiments.costs import PeakMemory, synchronized_time
from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.history_transfer.inference import joint_evidence
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.selection import select


def test_cpu_measurement_does_not_use_an_unrelated_cuda_context(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU task touched CUDA measurement")

    for name in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    with PeakMemory(device="cpu") as measured:
        assert synchronized_time("cpu") > 0
    assert measured.result["device"] == "cpu"
    assert measured.result["peak_cuda_allocated_bytes"] == 0
    assert measured.result["wall_s"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_joint_evidence_and_parameter_gradients_match_cpu_cuda():
    torch.set_num_threads(1)
    torch.manual_seed(1301)
    model = JointResponseKernel(3).double()
    other = copy.deepcopy(model).cuda()
    x = features(sample("car_following", 4, "new", replicate=103))
    labels, rss = [False, True, False, True], [-0.2, -0.8, 0.3, -0.9]
    cpu = joint_evidence(model, x, labels, rss)
    gpu = joint_evidence(other, x, labels, rss)
    cpu.backward()
    gpu.backward()
    torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-8, rtol=1e-8)
    for (name, left), (other_name, right) in zip(
        model.named_parameters(), other.named_parameters()
    ):
        assert name == other_name
        torch.testing.assert_close(left.grad, right.grad.cpu(), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("raw_only", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_online_cpu_cuda_probabilities_weights_and_selection_match(raw_only):
    torch.manual_seed(1301)
    scenes = sample("car_following", 10, "new", replicate=104)
    model = None if raw_only else JointResponseKernel(3).double()
    sources = (
        None
        if raw_only
        else {
            "names": ["history"],
            "weights": np.ones(1),
            "means": np.tile([-1.0, 0.2], (1, len(scenes), 1)),
            "blocks": np.tile([[0.4, -0.06], [-0.06, 0.2]], (1, len(scenes), 1, 1)),
        }
    )

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                index % 2 == 0,
                SafetyMeasurements(1, 1, rss_margin=-0.8 if index % 2 == 0 else 0.2),
                0,
                measurement_contract(),
            )

    results = [
        select(
            scenes,
            Oracle(),
            pretrained=model,
            sources=sources,
            initial_indices=[0, 1],
            budget=4,
            device=device,
            batch_size=3,
        )
        for device in ("cpu", "cuda")
    ]
    assert results[0]["selected_indices"] == results[1]["selected_indices"]
    for name in (
        "prequery_probability",
        "prequery_component_weights",
        "component_weights_after_query",
    ):
        np.testing.assert_allclose(
            [q[name] for q in results[0]["queries"]],
            [q[name] for q in results[1]["queries"]],
            atol=1e-7,
            rtol=1e-7,
        )
    assert results[0]["resources"]["peak_cuda_allocated_bytes"] == 0
    assert results[1]["resources"]["peak_cuda_allocated_bytes"] > 0
    assert results[1]["resources"]["device"] == "cuda"
