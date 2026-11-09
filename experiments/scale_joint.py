"""Vary one computational scale at a time using recorded responses and public coordinates."""

import copy
import multiprocessing
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from testing_methods.history_transfer.inference import (
    adapted_model,
    predict_components,
    source_tensors,
)
from testing_methods.history_transfer.model import JointResponseKernel

from .costs import PeakMemory, synchronized_time
from .history import build_source, load_training_archives
from .joint_validation import OUTPUT, initial_models
from .prepare_joint import DATASET_ROOT
from .scenarios import candidate_pool, features
from .storage import write_json

AXES = {
    "sources": (2, 4, 8, 16, 32),
    "source_records": (16, 32, 128, 256, 512),
    "candidates": (1000, 2000, 4000, 8000),
    "target_records": (20, 50, 100, 200),
}
BASE = {"sources": 8, "source_records": 128, "candidates": 2000, "target_records": 50}
MEASUREMENT_SCOPE = (
    "Fresh worker per device/axis/value, after one excluded warmup with the same settings; "
    "process startup and model loading are excluded. Host RSS includes the worker and its "
    "same-case allocator residency."
)


def benchmark(models, archives, *, settings, kind, batch_size=128):
    base, encoder = models
    target = archives[0]
    t = min(settings["target_records"], len(target.scenarios))
    if t != settings["target_records"]:
        raise ValueError("Target scale exceeds the available executed response records")
    candidates = [
        *target.scenarios[:t],
        *candidate_pool(kind, settings["candidates"] - t, replicate=30),
    ]
    sources = []
    for index in range(settings["sources"]):
        archive = archives[index % len(archives)]
        n = settings["source_records"]
        if len(archive.scenarios) < n:
            raise ValueError("Source scale exceeds available records")
        sources.append(
            replace(
                archive,
                source_id=f"{archive.source_id}:computational_replica:{index}",
                scenarios=archive.scenarios[:n],
                collision=archive.collision[:n],
                safety=archive.safety[:n],
                rss_margin=archive.continuous_margin[:n],
            )
        )
    x = features(candidates)
    device = base.mean.device
    labels, margins = target.collision[:t], target.continuous_margin[:t]
    with PeakMemory(device=device) as memory:
        started = synchronized_time(device)
        priors = build_source(encoder, sources, candidates, batch_size=batch_size)
        construction_s = synchronized_time(device) - started
        started = synchronized_time(device)
        model = adapted_model(base, x[:t], labels, margins)
        physical = adapted_model(
            JointResponseKernel(base.dimension, deep=False).double().to(device),
            x[:t],
            labels,
            margins,
            learning_rate=0.01,
        )
        adaptation_s = synchronized_time(device) - started
        started = synchronized_time(device)
        with torch.no_grad():
            components, _ = predict_components(
                model,
                physical,
                x,
                source_tensors(priors, model),
                base.point_mean(1)[0],
                list(range(t)),
                labels,
                margins,
                batch_size=batch_size,
            )
            weights = components.new_tensor(np.r_[0.9 * 0.75 * priors["weights"], 0.9 * 0.25, 0.1])
            probability = (weights[:, None] * components).sum(0)
            values = probability.detach().cpu().numpy()
            maximum = values[t:].max()
            selected = min(
                (i for i in range(t, len(candidates)) if values[i] >= maximum - 1e-12),
                key=lambda i: candidates[i].scenario_id,
            )
        prediction_s = synchronized_time(device) - started
    return {
        "settings": settings,
        "source_construction_s": construction_s,
        "adaptation_s": adaptation_s,
        "prediction_s": prediction_s,
        "selected_index": selected,
        "probabilities": values.tolist(),
        "resources": memory.result,
        "interpretation": "Computational replicas preserve recorded responses; this is not a transfer-accuracy experiment.",
    }


def measure_case(models, archives, *, settings, kind, device):
    """Warm and measure one scale in a worker with no earlier scale allocations."""
    torch.set_num_threads(1)
    current = tuple(copy.deepcopy(model).to(device) for model in models)
    warmup = benchmark(current, archives, settings=settings, kind=kind)
    warmup.pop("probabilities")
    measured = benchmark(current, archives, settings=settings, kind=kind)
    measured["worker_pid"] = warmup["worker_pid"] = os.getpid()
    return measured, warmup


def run(
    destination=OUTPUT / "scaling",
    *,
    kind="front_brake",
    axes=AXES,
    base_settings=BASE,
    models=None,
    archives=None,
    devices=("cpu",),
):
    torch.set_num_threads(1)
    models = initial_models(kind) if models is None else models
    if archives is None:
        archives = load_training_archives(DATASET_ROOT / "manifest.json", kind)
    rows, warmups, references = [], {}, {}
    hardware = {
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    for device in devices:
        warmups[device] = []
        for axis, values in axes.items():
            for value in values:
                settings = {**base_settings, axis: value}
                with ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    measured, warmup = pool.submit(
                        measure_case,
                        models,
                        archives,
                        settings=settings,
                        kind=kind,
                        device=device,
                    ).result()
                warmups[device].append({"axis": axis, "device": device, **warmup})
                probabilities = np.asarray(measured.pop("probabilities"))
                key = axis, value
                if key not in references:
                    references[key] = probabilities, measured["selected_index"]
                reference, selected = references[key]
                np.testing.assert_allclose(probabilities, reference, atol=1e-8, rtol=1e-7)
                if measured["selected_index"] != selected:
                    raise ValueError("Compute devices selected different candidates")
                rows.append(
                    {
                        "axis": axis,
                        "device": device,
                        **measured,
                        "reference_probability_max_error": float(
                            np.max(np.abs(probabilities - reference))
                        ),
                    }
                )
                write_json(
                    Path(destination) / "reports/scaling.json",
                    {
                        "status": "running",
                        "batch_size": 128,
                        "runs": rows,
                        "warmups": warmups,
                        "hardware": hardware,
                        "measurement_scope": MEASUREMENT_SCOPE,
                    },
                )
    result = {
        "status": "complete",
        "batch_size": 128,
        "runs": rows,
        "warmups": warmups,
        "hardware": hardware,
        "measurement_scope": MEASUREMENT_SCOPE,
    }
    write_json(Path(destination) / "reports/scaling.json", result)
    return result


if __name__ == "__main__":
    run(devices=("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
