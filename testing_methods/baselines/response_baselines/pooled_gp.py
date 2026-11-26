"""One joint GP conditioned on raw historical and disclosed target responses."""

import copy
import sys
import time
from contextlib import contextmanager

import numpy as np
import psutil
import torch

from experiments.costs import measure_resources, synchronized_time
from experiments.rss import measurement_contract
from experiments.scenarios import features
from experiments.storage import write_json
from testing_methods.common.classification import torch_moments
from testing_methods.history_transfer.adaptation import adapted_online
from testing_methods.history_transfer.inference import (
    TARGET_PARAMETERS,
    JointPosterior,
    adapted_model,
)
from testing_methods.selection import validate_initial_indices

from .grouped_gp import predict_grouped


@contextmanager
def profile_stages(path, *, device="cpu"):
    """Observe the resource gate without changing the numerical operations.

    Durations are inclusive: posterior construction contains RSS and Laplace;
    optimization contains the registered number of posterior evaluations.
    """
    if path is None:
        yield
        return
    process = psutil.Process()
    active, stages = {}, {}
    started = synchronized_time(device)

    def observe(frame, event, arg):
        name = frame.f_code.co_name
        filename = frame.f_code.co_filename.replace("\\", "/")
        stage = None
        if filename.endswith("/history_transfer/inference.py"):
            stage = {"fit_joint": "optimization_10_steps", "__init__": "rss_and_laplace"}.get(name)
        elif filename.endswith("/history_transfer/adaptation.py") and name == "adapted_online":
            stage = "configured_online_fit"
        elif filename.endswith("/response_baselines/grouped_gp.py") and name == "predict_grouped":
            stage = "grouped_fit_and_predict"
        elif filename.endswith("/common/gp_laplace.py") and name == "laplace":
            stage = "laplace"
        elif filename.endswith(
            ("/history_transfer/model.py", "/history_transfer/shared_private.py")
        ):
            stage = {"features": "features", "kernel_from_features": "kernel"}.get(name)
        elif filename.endswith("/response_baselines/pooled_gp.py") and name == "predict_pooled":
            stage = "fit_and_predict"
        if stage is None:
            return
        if event == "call":
            active[id(frame)] = synchronized_time(device)
        elif event == "return" and id(frame) in active:
            row = stages.setdefault(
                stage, {"calls": 0, "inclusive_wall_s": 0.0, "observed_rss_bytes": 0}
            )
            row["calls"] += 1
            row["inclusive_wall_s"] += synchronized_time(device) - active.pop(id(frame))
            row["observed_rss_bytes"] = max(row["observed_rss_bytes"], process.memory_info().rss)
            write_json(
                path,
                {
                    "status": "running",
                    "pid": process.pid,
                    "wall_s": time.perf_counter() - started,
                    "stages": stages,
                    "timing": "inclusive; overlapping stages must not be summed",
                },
            )

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)
        write_json(
            path,
            {
                "status": "ended",
                "pid": process.pid,
                "wall_s": synchronized_time(device) - started,
                "stages": stages,
                "timing": "inclusive; overlapping stages must not be summed",
            },
        )


def predict_pooled(
    base,
    train_x,
    labels,
    margins,
    query_x,
    *,
    batch_size=128,
    inference_backend="dense",
    fit_settings=None,
):
    """Restart the registered fit from the same frozen historical base."""
    if inference_backend == "grouped":
        options = (
            {}
            if fit_settings is None
            else {
                "steps": fit_settings.steps,
                "learning_rate": fit_settings.learning_rate,
                "trainable": fit_settings.parameters,
            }
        )
        return predict_grouped(
            base, train_x, labels, margins, query_x, batch_size=batch_size, **options
        )
    if inference_backend != "dense":
        raise ValueError("Unknown pooled GP inference backend")
    encoded = base.features(train_x).detach()
    model = (
        adapted_model(base, train_x, labels, margins, encoded=encoded)
        if fit_settings is None
        else adapted_online(base, train_x, labels, margins, fit_settings, encoded=encoded)
    )
    with torch.no_grad():
        state = JointPosterior(
            model.point_mean(len(train_x)), model.kernel_from_features(encoded), labels, margins
        )
        query = model.features(query_x)
        probabilities = []
        for start in range(0, len(query), batch_size):
            z = query[start : start + batch_size]
            mean, blocks = state.predict(
                model.point_mean(len(z)),
                model.kernel_from_features(z, encoded),
                model.point_covariance(len(z)),
            )
            probabilities.extend(
                torch_moments(mean[:, 0], blocks[:, 0, 0].clamp_min(1e-10))[0].cpu().tolist()
            )
    return (
        probabilities,
        {
            name: float(getattr(model, name))
            for name in (TARGET_PARAMETERS if fit_settings is None else fit_settings.parameters)
        },
        state.info,
    )


@measure_resources
def select(
    candidates,
    oracle,
    *,
    pretrained,
    history,
    query_order,
    budget=100,
    snapshot_counts=(0, 5, 10, 20, 50, 100),
    batch_size=128,
    feature_fn=features,
    checkpoint_path=None,
    profile_path=None,
    threads=1,
    device="cpu",
    inference_backend="dense",
    fit_settings=None,
):
    torch.set_num_threads(threads)
    order = list(query_order)
    if budget:
        validate_initial_indices(len(candidates), order, budget)
    if len(order) != budget or any(k < 0 or k > budget for k in snapshot_counts):
        raise ValueError("Pooled replay needs the complete registered query order and checkpoints")
    pretrained = copy.deepcopy(pretrained).double().to(device).eval()
    for parameter in pretrained.parameters():
        parameter.requires_grad_(False)
    x = feature_fn(candidates)
    history_x = np.asarray(history["x"], dtype=float)
    history_y = np.asarray(history["collision"], dtype=bool)
    history_r = np.asarray(history["rss_margin"], dtype=float)
    if history_x.shape != (len(history_y), x.shape[1]) or history_r.shape != history_y.shape:
        raise ValueError("Raw historical coordinates, collision labels and RSS must align")
    selected, labels, margins, queries, snapshots = [], [], [], [], []
    for count in range(budget + 1):
        if count in snapshot_counts:
            train_x = np.concatenate((history_x, x[selected]))
            train_y = np.concatenate((history_y, labels))
            train_r = np.concatenate((history_r, margins))
            with profile_stages(profile_path, device=device):
                probability, parameters, diagnostic = predict_pooled(
                    pretrained,
                    train_x,
                    train_y,
                    train_r,
                    x,
                    batch_size=batch_size,
                    inference_backend=inference_backend,
                    fit_settings=fit_settings,
                )
            snapshots.append(
                {
                    "paid_queries": count,
                    "observed_indices": list(selected),
                    "probabilities": probability,
                    "component_weights": [1.0],
                    "target_parameters": parameters,
                    "training_count": len(train_y),
                    "conditional_mode": diagnostic,
                }
            )
            if checkpoint_path is not None:
                write_json(
                    checkpoint_path, {"status": "running", "prediction_snapshots": snapshots}
                )
        if count == budget:
            break
        index = order[count]
        feedback = oracle.query(index, purpose="initial" if count == 0 else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("Pooled feedback belongs to another candidate")
        if feedback.safety.rss_margin is None or feedback.rss_contract != measurement_contract():
            raise ValueError("Pooled GP requires the registered joint feedback")
        selected.append(index)
        labels.append(feedback.collision)
        margins.append(feedback.safety.rss_margin)
        queries.append({"index": index, "query": count + 1, "feedback": feedback.to_dict()})
    return {
        "method": "pooled_history",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "prediction_snapshots": snapshots,
        "historical_count": len(history_y),
        "inference_backend": inference_backend,
        "fit_settings": None if fit_settings is None else fit_settings.to_dict(),
        "response_kernel": type(pretrained).__name__,
        "representation": "historical_embedding",
        "feedback_used": "joint",
        "torch_threads": torch.get_num_threads(),
        "conditional_inference_device": str(device),
        "kernel_and_adaptation_device": str(device),
        "termination": "budget_exhausted",
    }
