"""Require reference observation only when binary discovery loss is unresolved."""
import numpy as np

from .decision_support import discovery_loss_support


def run_supported_policy(reference, candidate, oracle, tie, budget, initialization):
    available = np.ones(len(reference.session.remaining), dtype=bool)
    known, selected, observations, checks, support_checks = {}, [], [], [], []
    found = 0
    rate = 1 / np.sqrt(budget)
    for t in range(budget):
        reference.advance(known)
        before = reference.certificate(known)
        head = int(reference.head())
        scheduled = t >= initialization and (t - initialization) % 2 == 0
        allowed = bool(before["negative_extra_count"] + 1 <= 1 + rate * found + 1e-12)
        index, proposed, support = head, head, None
        if t >= initialization and allowed:
            start = len(candidate.selection_checks)
            proposed = int(candidate.rank(available, tie_break=tie)[0])
            if proposed != head:
                support = discovery_loss_support(candidate, proposed, head)
            index = proposed if not scheduled or support is not None and support["supported"] else head
            for selection in candidate.selection_checks[start:]:
                selection.update(query=t + 1, proposed_index=proposed)
                if index == head:
                    selection.update(chosen_index=head, chosen_probability=selection["reference_probability"],
                                     component_chosen_probabilities=selection["component_reference_probabilities"])
        if not available[index]:
            raise ValueError("Supported correction repeats a paid query")
        support_checks.append({"query": t + 1, "reference_index": head, "proposed_index": proposed,
                               "chosen_index": index, "scheduled_reference": scheduled, "probe_credit_available": allowed,
                               "binary_support": support, "reference_observation_skipped": bool(scheduled and index != head)})
        observation = oracle.query(index)
        if observation.risk is None or observation.collision is None:
            raise ValueError("Supported correction requires the same real risk/event feedback")
        known[index] = float(observation.risk), bool(observation.collision)
        found += int(known[index][1])
        available[index] = False
        selected.append(index)
        observations.append({"index": index, "risk": known[index][0], "collision": known[index][1],
                             "model_feedback_used": True, "role": "initialization" if t < initialization else
                             "reference" if index == head else "alternative"})
        candidate.observe(index, *known[index])
        reference.advance(known)
        check = reference.certificate(known)
        check.update(query=t + 1, chosen_index=index, reference_index=head, found=found,
                     loss_credit=1 + rate * found, credit_before=1 + rate * (found - int(known[index][1])),
                     negative_extra_before=before["negative_extra_count"], probe_allowed=allowed,
                     credit_rate=float(rate), scheduled_reference=scheduled)
        if check["negative_extra_count"] > check["loss_credit"] + 1e-12:
            raise ValueError("Supported correction exceeded actual discovery credit")
        checks.append(check)
    return {"selected_indices": selected, "queries": observations,
            "coupling_checks": checks, "reference_prefix_indices": reference.order,
            "area_gain_lower": sum(c["discovery_gain_lower"] for c in checks),
            "terminal_gain_lower": checks[-1]["discovery_gain_lower"],
            "alternative_queries": sum(o["role"] == "alternative" for o in observations),
            "max_negative_extra_count": max(c["negative_extra_count"] for c in checks),
            "discovery_support_checks": support_checks,
            "reference_observations_skipped": sum(c["reference_observation_skipped"] for c in support_checks)}
