"""Reconstruct saved GP forecasts from paid responses without rerunning selection."""

import copy

import numpy as np
import torch

from testing_methods.baselines.response_baselines.grouped_gp import predict_grouped
from testing_methods.common.classification import torch_moments
from testing_methods.history_transfer.inference import (
    JointPosterior,
    predict_components,
    source_tensors,
)
from testing_methods.history_transfer.shared_private import SharedPrivateKernel

from .scenarios import features


def _parameters(model, values, expected):
    if set(values) != set(expected) or any(not np.isfinite(v) for v in values.values()):
        raise ValueError("Saved GP parameters differ from the declared fitting configuration")
    with torch.no_grad():
        for name, value in values.items():
            getattr(model, name).fill_(value)


def reconstruct(
    candidates, snapshot, records, margins, *, method, models, pooled, sources, settings
):
    count = snapshot["paid_queries"]
    if type(count) is not int or count < 0 or count > len(records) or len(margins) != len(records):
        raise ValueError("Prediction checkpoint exceeds its measured paid responses")
    paid = records[:count]
    selected = [r["index"] for r in paid]
    if len(set(selected)) != count or snapshot["observed_indices"] != selected:
        raise ValueError("Forecast uses another paid-feedback sequence")
    labels = [r["feedback"]["collision"] for r in paid]
    if any(type(v) is not bool for v in labels):
        raise ValueError("Only measured Boolean collisions enter prediction reconstruction")
    margins = np.asarray(margins, dtype=float)
    if not np.isfinite(margins).all():
        raise ValueError("Measured RSS must be finite")
    x = features(candidates)
    with torch.no_grad():
        if method == "target_only":
            if models != (None, None) or sources is not None or pooled is not None:
                raise ValueError("Target-only audit must not access historical inputs")
            model = SharedPrivateKernel(x.shape[1], deep=False).double().eval()
            _parameters(model, snapshot["target_parameters"], settings.online.raw.parameters)
            encoded = model.features(x)
            state = JointPosterior(
                model.point_mean(count),
                model.kernel_from_features(encoded[selected]),
                labels,
                margins[:count],
            )
            outputs = []
            for start in range(0, len(x), 257):
                query = encoded[start : start + 257]
                mean, blocks = state.predict(
                    model.point_mean(len(query)),
                    model.kernel_from_features(query, encoded[selected]),
                    model.point_covariance(len(query)),
                )
                outputs.extend(
                    torch_moments(mean[:, 0], blocks[:, 0, 0].clamp_min(1e-10))[0].tolist()
                )
            return np.asarray(outputs)
        if method == "pooled_history":
            base = copy.deepcopy(models[0]).double().eval()
            _parameters(base, snapshot["target_parameters"], settings.pooled.parameters)
            probability, _, _ = predict_grouped(
                base,
                np.concatenate((pooled["x"], x[selected])),
                np.concatenate((pooled["collision"], labels)),
                np.concatenate((pooled["rss_margin"], margins[:count])),
                x,
                steps=0,
                batch_size=257,
                trainable=settings.pooled.parameters,
            )
            return np.asarray(probability)
        if method != "full" or sources is None:
            raise ValueError("Unknown GP reconstruction method or missing historical priors")
        base = copy.deepcopy(models[0]).double().eval()
        reference = base.point_mean(1)[0].detach().clone()
        model = copy.deepcopy(base)
        _parameters(model, snapshot["target_parameters"], settings.online.learned.parameters)
        physical = type(base)(base.dimension, deep=False).double().eval()
        _parameters(physical, snapshot["physical_parameters"], settings.online.raw.parameters)
        components, _ = predict_components(
            model,
            physical,
            x,
            source_tensors(sources, base),
            reference,
            selected,
            labels,
            margins[:count],
            batch_size=257,
        )
        weights = np.asarray(snapshot["component_weights"])
        if (
            weights.shape != (len(sources["names"]) + 2,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Saved historical mixture weights differ from the declared components")
        return (torch.as_tensor(weights)[:, None] * components).sum(0).numpy()


def verify_forecasts(
    candidates, snapshots, records, margins, *, method, models, pooled, sources, settings
):
    errors = []
    for snapshot in snapshots:
        saved = np.asarray(snapshot["probabilities"])
        if (
            saved.shape != (len(candidates),)
            or not np.isfinite(saved).all()
            or ((saved < 0) | (saved > 1)).any()
        ):
            raise ValueError("Saved forecast is incomplete or nonfinite")
        actual = reconstruct(
            candidates,
            snapshot,
            records,
            margins,
            method=method,
            models=models,
            pooled=pooled,
            sources=sources,
            settings=settings,
        )
        error = float(np.max(np.abs(actual - saved)))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"Independently reconstructed {method} forecast differs by {error}")
        errors.append({"K": snapshot["paid_queries"], "maximum_probability_error": error})
    if not errors:
        raise ValueError("Forecast audit requires saved prediction checkpoints")
    return {
        "checkpoints": errors,
        "maximum_probability_error": max(e["maximum_probability_error"] for e in errors),
    }


def verify_weight_updates(result, records, sources):
    """Recompute the global Bayesian weight chain from recorded pre-query forecasts."""
    names = [*sources["names"], "independent_target", "physical_target"]
    prior = np.r_[0.675 * np.asarray(sources["weights"], dtype=float), 0.225, 0.1]
    if result["component_names"] != names or result["local_reliability"]:
        raise ValueError("Weight audit requires the registered global mixture components")
    if len(result["queries"]) != len(records):
        raise ValueError("Mixture weight chain differs from the paid sequence")
    snapshots = {row["paid_queries"]: row for row in result["prediction_snapshots"]}
    scores = np.log(prior)
    maximum_error = 0.0

    def compare(actual, expected):
        nonlocal maximum_error
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("Mixture weight state is incomplete or nonfinite")
        error = float(np.max(np.abs(actual - expected)))
        if error > 1e-10:
            raise ValueError(f"Independently reconstructed mixture weight state differs by {error}")
        maximum_error = max(maximum_error, error)

    compare(result["group_prior_weights"], prior)
    weights = prior / prior.sum()
    if 0 in snapshots:
        compare(snapshots[0]["component_weights"], weights)
    for count, (query, record) in enumerate(zip(result["queries"], records), 1):
        if query["index"] != record["index"]:
            raise ValueError("Mixture weight update uses another paid candidate")
        probability = np.asarray(query["prequery_component_probabilities"], dtype=float)
        if probability.shape != prior.shape or not np.isfinite(probability).all() or (
            (probability < 0) | (probability > 1)
        ).any():
            raise ValueError("Invalid pre-query component probabilities")
        compare(query["prequery_component_weights"], weights)
        compare([query["prequery_probability"]], [weights @ probability])
        if result["source_weight_updates"]:
            likelihood = probability if record["feedback"]["collision"] else 1 - probability
            likelihood = np.log(np.maximum(likelihood, 1e-15))
            compare(query["prequery_log_likelihood"], likelihood)
            scores += likelihood
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= weights.sum()
        elif query["prequery_log_likelihood"] is not None:
            raise ValueError("Frozen mixture unexpectedly records a weight update")
        compare(query["component_weights_after_query"], weights)
        if count in snapshots:
            compare(snapshots[count]["component_weights"], weights)
    compare([result["physical_mass"]], [weights[-1]])
    return {"queries": len(records), "maximum_weight_error": maximum_error}
