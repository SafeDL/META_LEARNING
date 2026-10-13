"""Remaining-budget set inclusion and correlated risk-threshold replacement."""
import numpy as np

from .config import SET_SAMPLES, THRESHOLD


def set_inclusion(draws, horizon):
    high = np.asarray(draws) > THRESHOLD
    positive = high.sum(axis=1, keepdims=True)
    negative = high.shape[1] - positive
    positive_credit = np.minimum(1., horizon / np.maximum(positive, 1))
    negative_credit = np.maximum(horizon - positive, 0) / np.maximum(negative, 1)
    return np.mean(np.where(high, positive_credit, negative_credit), axis=0)


def choose_candidate(model, remaining, gated, transfer, trust, horizon, seed, budget):
    available = np.flatnonzero(remaining)
    order = available[np.argsort(gated[1, available], kind="stable")]
    head = int(order[0])
    if trust == 0:
        return head, {"reason": "reference_fallback", "trust": trust}
    horizon = min(horizon, len(available))
    gate_prefix = order[:horizon]
    transfer_order = available[np.argsort(transfer[1, available], kind="stable")]
    union = np.union1d(gate_prefix, transfer_order[:horizon])
    draws = model.draws(union, SET_SAMPLES, np.random.default_rng(seed), trust)
    inclusion = set_inclusion(draws, horizon)
    proposed = int(union[np.lexsort((union, gated[1, union], -inclusion))[0]])
    record = {
        "reference_index": head, "proposed_index": proposed,
        "horizon": horizon, "union_size": len(union),
        "set_inclusion_probability": float(inclusion[np.flatnonzero(union == proposed)[0]]),
        "trust": trust,
    }
    if proposed in gate_prefix:
        return proposed, {**record, "reason": "current_prefix_reordering"}
    transfer_pair, reference_pair = model.discordance(proposed, head)
    gated_pair = trust * transfer_pair + (1 - trust) * reference_pair
    allowance = 1 / np.sqrt(budget)
    supported = transfer_pair[0] > transfer_pair[1] + 1e-12 and gated_pair[1] <= allowance
    return (proposed if supported else head), {
        **record, "reason": "supported_set_change" if supported else "replacement_rejected",
        "transfer_gain": float(transfer_pair[0]), "transfer_loss": float(transfer_pair[1]),
        "gated_loss": float(gated_pair[1]), "loss_allowance": float(allowance),
    }
