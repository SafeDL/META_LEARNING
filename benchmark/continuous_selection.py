"""Shared actual-query budget; each continuous algorithm retains its search mechanism."""

import time

import torch

from methods.baselines.deep_hyperion.search import DeepHyperion
from methods.baselines.indago.search import Indago
from methods.baselines.samota.search import Samota
from methods.gcrt.continuous import ContinuousDiscovery

from .continuous_space import SearchSpace
from .schema import QueryFeedback


METHODS = ("history_response", "indago", "samota", "deep_hyperion", "random")


def select(initial, oracle, *, method, seed, budget=128, history=(), encoder=None,
           classifier=None, target_weight=0.25, configuration=None, ordinary_only=False):
    if method not in METHODS or not 0 < len(initial) <= budget:
        raise ValueError("Invalid continuous method or common initialization budget")
    if len({s.coordinate() for s in initial}) != len(initial):
        raise ValueError("Common initialization must contain distinct physical coordinates")
    torch.set_num_threads(1)
    space = SearchSpace(initial[0].kind, seed + 1910003, ordinary_only=ordinary_only)
    if any(not space.valid(scene) for scene in initial):
        raise ValueError("Common initialization lies outside the registered search distribution")
    started = time.perf_counter()
    arguments = {} if configuration is None else {"config": configuration}
    if method == "history_response":
        search = ContinuousDiscovery(space, initial, encoder, history,
                                     target_weight=target_weight, **arguments)
    elif method == "indago":
        if classifier is None:
            raise ValueError("Indago requires its allowed historical classifier weights")
        search = Indago(space, classifier, **arguments)
    elif method == "samota":
        search = Samota(space, **arguments)
    elif method == "deep_hyperion":
        search = DeepHyperion(space, **arguments)
    else:
        search = None
    construction_s = time.perf_counter() - started
    queries, candidates = [], list(initial)
    for step in range(budget):
        started = time.perf_counter()
        diagnostics = {}
        if step < len(initial):
            scene, index = initial[step], step
            if method == "history_response":
                probability = search.probability(scene)
            elif method == "indago":
                probability = float(torch.sigmoid(torch.tensor(float(search.logits([scene])[0]))))
            else:
                probability = None
        else:
            if search is None:
                scene, probability = space.random(), None
            else:
                scene, probability, diagnostics = search.propose(budget - step)
            index = oracle.register(scene)
            if index != len(candidates):
                raise ValueError("Continuous selector and evaluator candidate registries diverged")
            candidates.append(scene)
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if not isinstance(feedback, QueryFeedback) or feedback.scenario_id != scene.scenario_id:
            raise ValueError("Continuous feedback does not match the paid proposal")
        space.remember(scene)
        started = time.perf_counter()
        adaptation = search.observe(scene, feedback) if search is not None else {}
        queries.append({"query": step + 1, "index": index, "feedback": feedback.to_dict(),
                        "prequery_probability": probability, "selection_s": selection_s,
                        "adaptation_s": time.perf_counter() - started, **diagnostics, **adaptation,
                        "reason": "common_initialization" if step < len(initial) else method})
    return {"method": method, "queries": queries, "actual_queries": len(queries),
            "maximum_queries": budget, "selected_indices": [q["index"] for q in queries],
            "model_construction_s": construction_s, "termination": "budget_exhausted"}
