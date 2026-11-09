"""Construct registered ADS controllers behind the shared driving interface."""

from .catalog import BY_ID
from .decision_interface import DecisionInterface
from .optimization import OptimizationController
from .rule import RuleController
from .search import SearchController


def create(sut_id):
    sut = BY_ID[sut_id]
    if sut.family == "rule":
        policy = RuleController(sut.lineage, sut.parameters)
    elif sut.family == "optimization":
        policy = OptimizationController(sut.lineage, sut.parameters)
    elif sut.family == "search":
        policy = SearchController(sut.lineage, sut.parameters)
    else:
        from .learning import LearnedController

        policy = LearnedController(sut)
    return DecisionInterface(policy)
