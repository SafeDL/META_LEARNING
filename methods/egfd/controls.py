"""Student-t controls for reference protection and risk scale uncertainty."""
import numpy as np


def run_policy(method, reference, candidate, oracle, tie, budget, initialization):
    available = np.ones(len(reference.session.remaining), dtype=bool)
    known, selected, observations, checks = {}, [], [], []
    found = 0
    credit_rate = 1 / np.sqrt(budget)
    for t in range(budget):
        reference.advance(known)
        before = reference.certificate(known)
        head = reference.head()
        credit_before = 1 + credit_rate * found
        probe_allowed = before["negative_extra_count"] + 1 <= credit_before + 1e-12
        if (t < initialization or method == "frozen_reference"
                or method == "coupled_response" and not probe_allowed):
            index = int(head)
        else:
            index = int(candidate.rank(available, tie_break=tie)[0])
        if not available[index]:
            raise ValueError("Duplicate paid query in frozen-reference coupling")
        observation = oracle.query(index)
        if observation.risk is None or observation.collision is None:
            raise ValueError("All frozen-reference mechanism controls require the same queried feedback")
        known[index] = (float(observation.risk), bool(observation.collision))
        found += int(known[index][1])
        selected.append(index)
        available[index] = False
        observations.append({"index": index, "risk": known[index][0], "collision": known[index][1],
                             "role": "initialization" if t < initialization else
                             "reference" if index == head else "alternative"})
        candidate.observe(index, *known[index])
        reference.advance(known)
        check = reference.certificate(known)
        check.update(query=t + 1, chosen_index=index, reference_index=head,
                     found=found, loss_credit=1 + credit_rate * found,
                     credit_before=credit_before, negative_extra_before=before["negative_extra_count"],
                     probe_allowed=bool(probe_allowed), credit_rate=float(credit_rate),
                     scope="whole frozen_original discovery curve; risk-only reference sees only its paid prefix")
        if method == "coupled_response" and check["negative_extra_count"] > check["loss_credit"] + 1e-12:
            raise ValueError("Frozen-reference coupling exceeds its negative-extra budget")
        checks.append(check)
    return {"selected_indices": selected, "queries": observations,
            "coupling_checks": checks, "reference_prefix_indices": reference.order,
            "area_gain_lower": sum(c["discovery_gain_lower"] for c in checks),
            "terminal_gain_lower": checks[-1]["discovery_gain_lower"],
            "alternative_queries": sum(o["role"] == "alternative" for o in observations),
            "max_negative_extra_count": max(c["negative_extra_count"] for c in checks),
            "credit_rate": float(credit_rate),
            "reference_feedback": "queried risk only, unchanged frozen implementation"}
