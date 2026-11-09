import os

import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scale_joint import run
from experiments.scenarios import sample
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.model import JointResponseKernel


@pytest.mark.parametrize(
    "devices",
    [
        ("cpu",),
        pytest.param(
            ("cpu", "cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_scaling_runner_changes_one_axis_and_reports_measured_resources(tmp_path, devices):
    scenes = sample("car_following", 8, "history")
    archive = ResponseArchive(
        "a",
        "rule",
        "a",
        "history",
        scenes,
        np.arange(8) % 3 == 0,
        np.ones((8, 2)),
        rss_margin=np.linspace(-0.8, 0.3, 8),
        rss_contract=measurement_contract(),
    )
    torch.manual_seed(1301)
    models = tuple(JointResponseKernel(3).double() for _ in range(2))
    result = run(
        tmp_path,
        kind="car_following",
        models=models,
        archives=[archive],
        axes={"candidates": (8, 16)},
        base_settings={"sources": 2, "source_records": 4, "candidates": 8, "target_records": 2},
        devices=devices,
    )
    assert result["status"] == "complete"
    assert [r["settings"]["candidates"] for r in result["runs"]] == [8, 16] * len(devices)
    assert set(result["warmups"]) == set(devices)
    assert "Fresh worker" in result["measurement_scope"]
    assert result["hardware"]["threads"] == 1
    assert len({row["worker_pid"] for row in result["runs"]}) == len(result["runs"])
    for device in devices:
        rows = [r for r in result["runs"] if r["device"] == device]
        warmups = result["warmups"][device]
        assert len(warmups) == len(rows) == 2
        for row, warmup in zip(rows, warmups, strict=True):
            assert row["worker_pid"] == warmup["worker_pid"] != os.getpid()
            assert row["settings"] == warmup["settings"]
            assert row["selected_index"] == warmup["selected_index"]
    for row in result["runs"]:
        assert row["resources"]["sampled_peak_cpu_bytes"] > 0
        assert row["prediction_s"] > 0 and row["adaptation_s"] > 0
        assert row["reference_probability_max_error"] < 1e-7
        if row["device"] == "cuda":
            assert row["resources"]["peak_cuda_allocated_bytes"] > 0
