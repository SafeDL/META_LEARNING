"""Opt-in joint-feedback discovery with spatial reliability and Risk-UCB."""

import copy

import numpy as np
import torch

from experiments.costs import measure_resources, synchronized_time
from experiments.domain import Domain
from experiments.rss import measurement_contract
from experiments.scenarios import features
from testing_methods.selection import validate_initial_indices

from .adaptation import adapted_online
from .inference import JointPosterior, JointPrior, initial_weights, source_tensors
from .lifelong_reliability import (
    TEMPERATURE, effective_weights, logistic_moments, mixture_moments, risk_beta,
    target_floor, update_scores,
)

CONFIGURATIONS = {
    "local_reliability": (True, False),
    "risk_ucb": (False, True),
    "lifelong_full": (True, True),
}


def predict_moments(priors, embeddings, paid, labels, margins, indices, batch_size):
    observed = {
        id(model): model.kernel_from_features(z[paid])
        for model, z in embeddings
    }
    encoded = {id(model): z for model, z in embeddings}
    states = [
        JointPosterior(
            prior.mean(paid),
            prior.covariance(paid, paid, base_covariance=observed[id(prior.model)]),
            labels, margins,
        )
        for prior in priors
    ]
    outputs = [[] for _ in priors]
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        cross = {
            id(model): model.kernel_from_features(z[batch], encoded[id(model)][paid])
            for model, z in embeddings
        }
        for output, prior, state in zip(outputs, priors, states, strict=True):
            mean, block = state.predict(
                prior.mean(batch),
                prior.covariance(batch, paid, base_covariance=cross[id(prior.model)]),
                prior.blocks(batch),
            )
            output.append(torch.stack(logistic_moments(mean[:, 0], block[:, 0, 0])))
    moments = torch.stack([torch.cat(output, dim=1) for output in outputs])
    return moments[:, 0], moments[:, 1], [state.info for state in states]


@measure_resources
def select(
    candidates, oracle, *, pretrained, sources, initial_indices, online_settings,
    configuration="lifelong_full", budget=200, device="cpu", batch_size=128,
    snapshot_counts=(0, 20, 50, 100, 200), query_order=None, update_source_weights=True,
    implicit_evidence=False,
):
    spatial, ucb = CONFIGURATIONS[configuration]
    if batch_size < 1 or not sources["names"] or (spatial and not update_source_weights):
        raise ValueError("Spatial reliability requires explicit sources and feedback weight updates")
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget)
    if query_order is not None:
        query_order = list(query_order)
        validate_initial_indices(len(candidates), query_order, budget)
        if len(query_order) != budget or query_order[:len(initial)] != initial:
            raise ValueError("Prescribed feedback order must match the budget and initialization")
    weights = np.asarray(sources["weights"])
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Historical sources require a normalized positive prior")
    definition = getattr(pretrained, "construction_specification", {}).get("parameter_definition")
    if definition != sources["specification"].get("parameter_definition"):
        raise ValueError("Source and target feature domains differ")
    x = features(candidates, domain=None if definition is None else Domain.from_dict(definition))
    torch.set_num_threads(1)
    base = copy.deepcopy(pretrained).to(device).double().eval()
    raw_base = type(base)(base.dimension, deep=False).to(device).double().eval()
    with torch.no_grad():
        raw_base.mean.fill_(online_settings.raw_initial_mean)
    for branch in (base, raw_base):
        branch.requires_grad_(False)
    embedded = base.features(x).detach()
    raw_embedded = raw_base.features(x).detach()
    reference = base.point_mean(1)[0].detach().clone()
    historical = source_tensors(sources, base)
    original = initial_weights(sources, reference)
    scores = original.log()
    scale = base.log_lengthscale.exp().clamp_min(0.01).detach()
    model, raw = base, raw_base
    paid, responses, margins, likelihoods, queries, snapshots = [], [], [], [], [], []
    full_indices = list(range(len(candidates)))
    remaining = set(full_indices)

    def global_weights():
        return target_floor(scores / TEMPERATURE) if spatial else scores.softmax(0)

    @torch.no_grad()
    def forecast(indices):
        priors = [JointPrior(model, x, reference, item) for item in historical]
        priors.extend((JointPrior(model, x, reference), JointPrior(raw, x, reference)))
        means, variances, diagnostic = predict_moments(
            priors, ((model, embedded), (raw, raw_embedded)),
            paid, responses, margins, indices, batch_size,
        )
        weight = effective_weights(
            scores, embedded[indices], embedded[paid], likelihoods, scale,
            spatial=spatial, batch_size=batch_size,
        )
        probability, epistemic = mixture_moments(weight, means, variances)
        return probability, epistemic, means, variances, weight, diagnostic

    def snapshot(step, forecast_values):
        probability, epistemic, _, _, weight, _ = forecast_values
        snapshots.append({
            "paid_queries": step, "observed_indices": list(paid),
            "probabilities": probability.cpu().tolist(),
            "epistemic_standard_deviations": epistemic.cpu().tolist(),
            "component_weights": weight.cpu().tolist() if spatial else weight[:, 0].cpu().tolist(),
            "target_parameters": {
                key: float(getattr(model, key)) for key in online_settings.learned.parameters
            },
            "physical_parameters": {
                key: float(getattr(raw, key)) for key in online_settings.raw.parameters
            },
        })

    for step in range(budget):
        started = synchronized_time(device)
        prescribed = query_order[step] if query_order is not None else (
            initial[step] if step < len(initial) else None
        )
        full_prediction = prescribed is None or step in snapshot_counts
        predicted = forecast(full_indices if full_prediction else [prescribed])
        probability, epistemic, means, variances, weights, diagnostics = predicted
        if step in snapshot_counts:
            snapshot(step, predicted)
        historical_mass = float(global_weights()[:-2].sum().clamp(0, 1))
        beta = risk_beta(step + 1, budget, len(initial), historical_mass) if (
            ucb and step >= len(initial)
        ) else 0.0
        acquisition = (probability + beta * epistemic).cpu().numpy()
        if not np.isfinite(acquisition).all():
            raise FloatingPointError("Nonfinite lifelong acquisition")
        if prescribed is None:
            maximum = max(acquisition[i] for i in remaining)
            index = min(
                (i for i in remaining if acquisition[i] >= maximum - 1e-12),
                key=lambda i: candidates[i].scenario_id,
            )
        else:
            index = prescribed
        column = index if full_prediction else 0
        selection_s = synchronized_time(device) - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if (
            feedback.scenario_id != candidates[index].scenario_id
            or feedback.rss_contract != measurement_contract()
            or feedback.safety.rss_margin is None
        ):
            raise ValueError("Lifelong discovery requires matching paid joint feedback")
        started = synchronized_time(device)
        paid.append(index)
        remaining.remove(index)
        responses.append(feedback.collision)
        margins.append(feedback.safety.rss_margin)
        likelihood = None
        if update_source_weights:
            scores, likelihood = update_scores(
                scores, means[:, column], feedback.collision, tempered=spatial,
            )
            if spatial:
                likelihoods.append(likelihood)
        model = adapted_online(
            base, x[paid], responses, margins, online_settings.learned, encoded=embedded[paid],
            implicit_evidence=implicit_evidence,
        )
        raw = adapted_online(
            raw_base, x[paid], responses, margins, online_settings.raw, encoded=raw_embedded[paid],
            implicit_evidence=implicit_evidence,
        )
        queries.append({
            "query": step + 1, "index": index, "feedback": feedback.to_dict(),
            "prequery_probability": float(probability[column]),
            "prequery_component_probabilities": means[:, column].cpu().tolist(),
            "prequery_component_weights": weights[:, column].cpu().tolist(),
            "prequery_component_epistemic_variances": variances[:, column].cpu().tolist(),
            "prequery_epistemic_standard_deviation": float(epistemic[column]),
            "prequery_historical_weight": historical_mass, "risk_beta": beta,
            "prequery_acquisition": float(acquisition[column]),
            "prequery_log_likelihood": None if likelihood is None else likelihood.cpu().tolist(),
            "component_weights_after_query": global_weights().cpu().tolist(),
            "conditional_modes": diagnostics[:-1], "physical_mode": diagnostics[-1],
            "conditioning_queries": step, "selection_s": selection_s,
            "adaptation_s": synchronized_time(device) - started,
            "reason": "common_initialization" if step < len(initial) else configuration,
        })
        if getattr(oracle, "stopped", False):
            break
    if len(paid) in snapshot_counts:
        snapshot(len(paid), forecast(full_indices))
    return {
        "method": configuration, "response_kernel": type(base).__name__,
        "queries": queries, "selected_indices": paid, "actual_queries": len(paid),
        "maximum_queries": budget,
        "termination": "evaluator_stop" if getattr(oracle, "stopped", False) else "budget_exhausted",
        "group_prior_weights": original.cpu().tolist(),
        "component_names": [*sources["names"], "independent_target", "physical_target"],
        "prediction_snapshots": snapshots, "feedback_used": "joint",
        "local_reliability": spatial, "uncertainty_acquisition": ucb,
        "target_feedback_updates": True, "source_weight_updates": update_source_weights,
        "conditional_inference_device": device, "online_settings": online_settings.to_dict(),
    }
