"""Paid-feedback replay of the exact frozen risk-only adaptive reference."""
import numpy as np
from threadpoolctl import threadpool_limits

from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.search import TestingSession

from .config import BUDGET, INITIAL_QUERIES
from .risk_event_response import RiskEventResponse
from .student_risk_prior import add_student_risk_variation


METHODS = ("frozen_reference", "uncoupled_response", "coupled_response")


class FrozenReplay:
    def __init__(self, session):
        self.session = session
        self.order = []

    def head(self):
        if self.session.pending is None:
            return self.session.next_index()
        return int(self.session.pending)

    def advance(self, known):
        while (index := self.head()) is not None and index in known:
            self.session.observe(known[index][0])
            self.order.append(index)

    def certificate(self, known):
        extra = [index for index in known if self.session.remaining[index]]
        negative = [index for index in extra if not known[index][1]]
        return {"reference_prefix_length": len(self.order), "extra_indices": extra,
                "negative_extra_indices": negative, "negative_extra_count": len(negative),
                "discovery_gain_lower": -len(negative)}


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


def selector_worker(connection):
    try:
        with threadpool_limits(limits=1):
            while True:
                message, task = connection.recv()
                if message == "stop":
                    return
                if message != "task" or task["method"] not in METHODS:
                    raise ValueError("Invalid frozen-reference coupling task")
                session = TestingSession(task["x"], task["seed"], mode="global_feedback", prior=task["prior"])
                reference = FrozenReplay(session)
                candidate = RiskEventResponse(reference, task["x"][:, 4], task["event_readout"])
                candidate = add_student_risk_variation(candidate, task["risk_templates"])
                tie = np.random.default_rng(task["seed"]).random(len(task["x"]))
                result = run_policy(task["method"], reference, candidate, RemoteOracle(connection),
                                    tie, BUDGET, INITIAL_QUERIES)
                result.update(candidate.diagnostics())
                connection.send(("result", result))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()
