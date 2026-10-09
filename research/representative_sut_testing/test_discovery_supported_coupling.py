"""Unresolved reference comparisons and model-independent discovery credit."""
import itertools

import numpy as np

from . import discovery_supported_coupling as policy
from .frozen_coupling import FrozenReplay
from .test_frozen_coupling import Candidate, Oracle, ReferenceSession


def test_unresolved_comparison_keeps_reference_observation(monkeypatch):
    monkeypatch.setattr(policy, 'discovery_loss_support', lambda *args: {'supported': False})
    run = policy.run_supported_policy(FrozenReplay(ReferenceSession()), Candidate(), Oracle([True] * 5), np.arange(5), 5, 1)
    assert all(not c['reference_observation_skipped'] for c in run['discovery_support_checks'])
    for check in run['discovery_support_checks']:
        if check['scheduled_reference']:
            assert check['chosen_index'] == check['reference_index']


def test_supported_skips_preserve_full_reference_bounds_on_all_binary_banks(monkeypatch):
    monkeypatch.setattr(policy, 'discovery_loss_support', lambda *args: {'supported': True})
    skips = 0
    for labels in itertools.product((False, True), repeat=5):
        standalone = ReferenceSession()
        reference_order = []
        for _ in range(4):
            index = standalone.next_index()
            reference_order.append(index)
            standalone.observe(float(labels[index]))
        oracle = Oracle(labels)
        run = policy.run_supported_policy(FrozenReplay(ReferenceSession()), Candidate(), oracle, np.arange(5), 4, 1)
        actual = np.cumsum([labels[i] for i in oracle.paid])
        reference = np.cumsum([labels[i] for i in reference_order])
        assert len(set(oracle.paid)) == 4
        assert np.all(actual - reference >= [c['discovery_gain_lower'] for c in run['coupling_checks']])
        assert np.all(actual >= (reference - 1) / (1 + 1 / np.sqrt(4)))
        skips += run['reference_observations_skipped']
    assert skips > 0
