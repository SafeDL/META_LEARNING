"""Pending-query replay preserves an adaptive reference and full-curve bounds."""
import itertools
from types import SimpleNamespace

import numpy as np

from .frozen_coupling import FrozenReplay, run_policy


class ReferenceSession:
    def __init__(self):
        self.pending = None
        self.remaining = np.ones(5, dtype=bool)
        self.positive = False

    def next_index(self):
        assert self.pending is None
        order = [0, 2, 4, 1, 3] if self.positive else [0, 1, 3, 2, 4]
        choices = [index for index in order if self.remaining[index]]
        self.pending = choices[0] if choices else None
        return self.pending

    def observe(self, risk):
        assert self.pending is not None
        self.remaining[self.pending] = False
        self.pending = None
        self.positive = risk > .5


class Candidate:
    def __init__(self):
        self.positive = False
        self.selection_checks = []

    def rank(self, remaining, *, tie_break):
        indices = np.flatnonzero(remaining)
        return indices if self.positive else indices[::-1]

    def observe(self, index, risk, collision):
        self.positive = collision


class Oracle:
    def __init__(self, labels, risks=None):
        self.labels = labels
        self.risks = labels if risks is None else risks
        self.paid = []

    def query(self, index):
        assert index not in self.paid
        self.paid.append(index)
        return SimpleNamespace(risk=float(self.risks[index]), collision=self.labels[index])


def test_pending_reference_query_is_not_requested_twice():
    replay = FrozenReplay(ReferenceSession())
    assert replay.head() == replay.head() == 0
    replay.advance({0: (0., False)})
    assert replay.order == [0] and replay.head() == 1


def test_all_binary_banks_preserve_adaptive_reference_and_full_curve_bound():
    for labels in itertools.product((False, True), repeat=5):
        standalone = ReferenceSession()
        reference_order = []
        for _ in range(4):
            index = standalone.next_index()
            reference_order.append(index)
            standalone.observe(float(labels[index]))
        ref_curve = np.cumsum([labels[index] for index in reference_order])
        for method in ("frozen_reference", "uncoupled_response", "coupled_response"):
            oracle = Oracle(labels)
            run = run_policy(method, FrozenReplay(ReferenceSession()), Candidate(), oracle, np.arange(5), 4, 1)
            assert len(oracle.paid) == 4 and oracle.paid == run["selected_indices"]
            k = len(run["reference_prefix_indices"])
            assert run["reference_prefix_indices"] == reference_order[:k]
            gain = np.cumsum([labels[index] for index in oracle.paid]) - ref_curve
            bound = [check["discovery_gain_lower"] for check in run["coupling_checks"]]
            assert np.all(gain >= bound) and gain.sum() >= run["area_gain_lower"]
            if method == "coupled_response":
                found = np.cumsum([labels[index] for index in oracle.paid])
                assert np.all(found >= (ref_curve - 1) / (1 + 1 / np.sqrt(4)))
                for check in run["coupling_checks"]:
                    assert check["negative_extra_count"] <= 1 + check["found"] / np.sqrt(4)
            if method == "frozen_reference":
                assert oracle.paid == reference_order


def test_known_reference_replay_consumes_no_new_target_query():
    oracle = Oracle([False, True, True, True, False])
    run = run_policy("coupled_response", FrozenReplay(ReferenceSession()), Candidate(), oracle, np.arange(5), 5, 1)
    assert len(oracle.paid) == len(run["reference_prefix_indices"]) == 5
    assert run["terminal_gain_lower"] == 0


def test_relative_bound_does_not_require_risk_to_determine_collision():
    for risks in itertools.product((0., 1.), repeat=5):
        standalone = ReferenceSession()
        reference_order = []
        for _ in range(4):
            index = standalone.next_index()
            reference_order.append(index)
            standalone.observe(risks[index])
        for labels in itertools.product((False, True), repeat=5):
            oracle = Oracle(labels, risks)
            run = run_policy("coupled_response", FrozenReplay(ReferenceSession()), Candidate(),
                             oracle, np.arange(5), 4, 1)
            prefix = run["reference_prefix_indices"]
            assert prefix == reference_order[:len(prefix)]
            actual = np.cumsum([labels[index] for index in oracle.paid])
            reference = np.cumsum([labels[index] for index in reference_order])
            assert np.all(actual >= (reference - 1) / (1 + 1 / np.sqrt(4)))
            assert np.all(actual - reference >= [c["discovery_gain_lower"] for c in run["coupling_checks"]])


def test_paid_discoveries_earn_credit_and_exhausted_credit_returns_to_reference():
    class FixedReference(ReferenceSession):
        def __init__(self):
            super().__init__()
            self.remaining = np.ones(12, dtype=bool)

        def next_index(self):
            assert self.pending is None
            choices = np.flatnonzero(self.remaining)
            self.pending = int(choices[0]) if len(choices) else None
            return self.pending

    class ReverseCandidate(Candidate):
        def rank(self, remaining, *, tie_break):
            return np.flatnonzero(remaining)[::-1]

    labels = [True] * 7 + [False] * 5
    oracle = Oracle(labels)
    run = run_policy("coupled_response", FrozenReplay(FixedReference()), ReverseCandidate(),
                     oracle, np.arange(12), 9, 1)
    assert oracle.paid == [0, 11, 1, 2, 10, 3, 4, 5, 9]
    assert run["max_negative_extra_count"] == 3
    actual = np.cumsum([labels[index] for index in oracle.paid])
    reference = np.cumsum(labels[:9])
    assert np.all(actual >= (reference - 1) / (1 + 1 / np.sqrt(9)))
    for check in run["coupling_checks"]:
        if check["chosen_index"] != check["reference_index"]:
            assert check["probe_allowed"]
            assert check["negative_extra_before"] + 1 <= check["credit_before"] + 1e-12
