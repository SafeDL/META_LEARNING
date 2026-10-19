"""Direct conditional lookahead; stopping never changes the full query order."""

import numpy as np
import torch
from scipy.special import logsumexp

from methods.baselines.dart.planning import top_sum
from methods.common.config import JITTER, NOISE

from .config import BUDGET, FANTASIES, SEGMENT, TOLERANCE
from .posterior import JointPosterior


def shortlist(model, remaining):
    indices = np.flatnonzero(remaining)
    components = model.component_probabilities(indices)
    q = components @ model.weights
    history_weights = np.exp(model.log_weight[:-1] - logsumexp(model.log_weight[:-1]))
    history = components[:, :-1] @ history_weights
    criteria = (
        q,
        -(
            q * np.log(np.maximum(q, 1e-300))
            + (1 - q) * np.log(np.maximum(1 - q, 1e-300))
        ),
        np.abs(history - components[:, -1]),
    )
    return np.unique(
        np.concatenate(
            [indices[np.argsort(-score, kind="stable")[:8]] for score in criteria]
        )
    )


def fixed_sets(q, candidates, available, count):
    ordered = available[np.argsort(-q[available], kind="stable")]
    return np.asarray([ordered[ordered != x][:count] for x in candidates])


def acquisition_values(
    model, candidates, remaining, horizon, seed, *, gpu=None, analytic=True
):
    q = model.probabilities()
    if horizon == 1:
        return q[candidates].copy()
    draws = model.marginal_draws(candidates, FANTASIES, np.random.default_rng(seed))
    available = np.flatnonzero(remaining)
    gpu = torch.cuda.is_available() if gpu is None else gpu
    if gpu:
        return gpu_acquisition(
            model, candidates, available, horizon, draws, q, analytic
        )
    values = []
    sets = fixed_sets(q, candidates, available, horizon - 1)
    for column, index in enumerate(candidates):
        future = available[available != index]
        updated = model.conditional_probabilities(index, draws[:, column], future)
        future_sum = top_sum(updated, horizon - 1)
        if analytic:
            positions = np.searchsorted(future, sets[column])
            adjustment = future_sum - updated[:, positions].sum(-1)
            if adjustment.min() < -1e-10:
                raise ArithmeticError("A fixed successor set exceeds the top set")
            value = q[index] + q[sets[column]].sum() + adjustment.mean()
        else:
            value = q[index] + future_sum.mean()
        values.append(value)
    return np.asarray(values)


def gpu_acquisition(model, candidates, available, horizon, draws, q, analytic):
    import cupy as cp
    from methods.common.student import student_cdf
    from cupyx.scipy.special import gammaln, logsumexp

    output = q[candidates].copy()
    sets = fixed_sets(q, candidates, available, horizon - 1)
    alpha = model.groups[0][1].alpha
    for start in range(0, len(candidates), 8):
        selected = candidates[start : start + 8]
        virtual = cp.asarray(draws[:, start : start + 8].T)
        means = cp.empty((len(selected), len(draws), len(available), model.count))
        variance = cp.empty((len(selected), len(available), model.count))
        beta = cp.empty((len(selected), len(draws), model.count))
        density = cp.empty_like(beta)
        for columns, component in model.groups:
            col = cp.asarray(columns)
            denominator = cp.asarray(component.variance[selected] + NOISE + JITTER)
            cross = cp.asarray(component.covariance[np.ix_(selected, available)])
            cross -= (
                cp.asarray(component.solved[selected])
                @ cp.asarray(component.solved[available]).T
            )
            residual = (
                virtual[:, :, None] - cp.asarray(component.mean[selected])[:, None]
            )
            means[..., col] = (
                cp.asarray(component.mean[available])[None, None]
                + cross[:, None, :, None]
                * residual[:, :, None]
                / denominator[:, None, None, None]
            )
            variance[..., col] = (
                cp.asarray(component.variance[available])[None, :, None]
                + NOISE
                + JITTER
                - cross[:, :, None] ** 2 / denominator[:, None, None]
            )
            scale = cp.asarray(component.beta)[None, None]
            beta[..., col] = scale + residual**2 / (2 * denominator[:, None, None])
            working = scale / alpha * denominator[:, None, None]
            density[..., col] = (
                gammaln((2 * alpha + 1) / 2)
                - gammaln(alpha)
                - 0.5 * cp.log(2 * alpha * cp.pi * working)
                - (2 * alpha + 1) / 2 * cp.log1p(residual**2 / (2 * alpha * working))
            )
        weights = density + cp.asarray(model.log_weight)
        weights -= logsumexp(weights, axis=-1, keepdims=True)
        z = (means - 0.5) / cp.sqrt(
            variance[:, None] * beta[:, :, None] / (alpha + 0.5)
        )
        updated = (student_cdf(2 * alpha + 1, z) * cp.exp(weights[:, :, None])).sum(-1)
        updated = cp.where(
            cp.asarray(selected[:, None, None] == available[None, None, :]),
            -cp.inf,
            updated,
        )
        updated = cp.asnumpy(updated)
        future_sum = top_sum(updated, horizon - 1)
        if analytic:
            fixed = sets[start : start + len(selected)]
            positions = np.searchsorted(available, fixed)
            fixed_sum = np.take_along_axis(updated, positions[:, None, :], axis=2).sum(
                -1
            )
            adjustment = future_sum - fixed_sum
            if adjustment.min() < -1e-10:
                raise ArithmeticError("A fixed successor set exceeds the top set")
            output[start : start + len(selected)] += q[fixed].sum(-1) + adjustment.mean(
                -1
            )
        else:
            output[start : start + len(selected)] += future_sum.mean(-1)
    return output


def choose(model, remaining, horizon, seed, *, analytic=True):
    candidates = shortlist(model, remaining)
    q = model.probabilities()
    values = acquisition_values(
        model, candidates, remaining, horizon, seed, analytic=analytic
    )
    tied = np.flatnonzero(values >= values.max() - TOLERANCE)
    best = tied[np.lexsort((candidates[tied], -q[candidates[tied]]))[0]]
    available = np.flatnonzero(remaining)
    greedy_index = int(available[np.argmax(q[available])])
    static_value = float(top_sum(q[available], horizon))
    return int(candidates[best]), {
        "candidate_indices": candidates.tolist(),
        "candidate_values": values.tolist(),
        "selected_value": float(values[best]),
        "horizon": int(horizon),
        "static_value": static_value,
        "value_over_static": float(values[best] - static_value),
        "marginal_best_index": greedy_index,
        "differs_from_marginal": int(candidates[best]) != greedy_index,
    }


def select(
    prior,
    oracle,
    *,
    seed=11,
    budget=BUDGET,
    initial_indices,
    stopper=None,
    recorder=None,
    clock=None,
    kappa=1.0,
    target_probability=0.25,
    policy="segment",
):
    initial_indices = list(map(int, initial_indices))
    if policy not in ("remaining", "analytic", "segment"):
        raise ValueError(f"Unknown selection policy: {policy}")
    if not (
        len(set(initial_indices)) == len(initial_indices)
        and 0 <= len(initial_indices) <= budget <= len(prior.x)
        and all(0 <= i < len(prior.x) for i in initial_indices)
    ):
        raise ValueError("Initialization must be unique and included in the budget")
    if stopper is not None:
        stopper.validate(prior, budget, initial_indices, policy)
    model = JointPosterior(prior, kappa, target_probability)
    remaining = np.ones(len(prior.x), dtype=bool)
    queries, decisions, evidence, stopping = [], [], [], []
    if clock is not None:
        clock.start()
    for step in range(budget):
        context = dict(
            model=model,
            remaining=remaining,
            queries=queries,
            evidence=evidence,
            budget=budget,
            initial_indices=initial_indices,
        )
        if stopper is not None and step >= len(initial_indices) and step % SEGMENT == 0:
            check = stopper.evaluate(**context)
            stopping.append(check)
            if check["stop"]:
                if clock is not None:
                    clock.finish_decision()
                break
        if step < len(initial_indices):
            index = initial_indices[step]
            decision = {"reason": "common_initialization", "horizon": budget - step}
        else:
            horizon = (
                min(SEGMENT - step % SEGMENT, budget - step)
                if policy == "segment"
                else budget - step
            )
            index, decision = choose(
                model,
                remaining,
                horizon,
                29011 + seed + step,
                analytic=policy != "remaining",
            )
            decision["reason"] = "conditional_lookahead"
        if recorder is not None:
            recorder.before_query(index=index, **context)
        if stopper is not None:
            stopper.before_query(index=index, **context)
        probability = float(model.probabilities()[index])
        risk = float(oracle.query(index).risk)
        model.observe(index, risk)
        remaining[index] = False
        queries.append({"index": index, "risk": risk, "high_risk": risk > 0.5})
        evidence.append({"prequery_probability": probability})
        decisions.append(
            {
                "query": step + 1,
                "selected_index": index,
                "source_weights": model.weights.tolist(),
                **decision,
            }
        )
        if stopper is not None:
            stopper.observe(risk, step + 1)
        if clock is not None:
            clock.step()
    return {
        "selected_indices": [q["index"] for q in queries],
        "queries": queries,
        "decisions": decisions,
        "evidence": evidence,
        "stopping": stopping,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "stop_reason": (
            "quality_retained" if len(queries) < budget else "budget_exhausted"
        ),
        "source_names": [*prior.names, "target_only"],
        "feedback_used": "queried continuous risk only",
        "policy": policy,
    }
