"""A low-information event model cannot erase reference regime exploration."""
from types import SimpleNamespace

import numpy as np

from .risk_event_response import RiskEventResponse


def toy(probability):
    model = object.__new__(RiskEventResponse)
    model.reference = SimpleNamespace(head=lambda: 1)
    model.family = np.asarray([0, 1, 1])
    model.gp = SimpleNamespace(mean=np.asarray([.9, .2, .7]))
    model.indices = []
    model.selection_checks = []
    model.probabilities = lambda: np.asarray(probability)
    return model


def test_reference_regime_is_retained_when_other_regime_has_higher_probability():
    model = toy([.99, .1, .2])
    assert model.rank(np.ones(3, dtype=bool), tie_break=np.arange(3))[0] == 2
    assert model.selection_checks[-1]['family'] == 1


def test_equal_probability_keeps_reference_instead_of_risk_tie_break():
    model = toy([.99, .01, .01])
    assert model.rank(np.ones(3, dtype=bool), tie_break=np.arange(3))[0] == 1
