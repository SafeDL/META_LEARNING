"""Fixed-classifier, saliency-guided hill climbing with public pool projection."""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from experiments.scenarios import features
from testing_methods.selection import validate_initial_indices

from .mutation import SearchSpace


@dataclass(frozen=True)
class SearchConfig:
    neighborhood_size: int = 50
    max_neighborhoods: int = 100
    max_stagnation: int = 20
    stagnation_tolerance: float = 0.005


DEFAULT = SearchConfig()


class PoolSearch:
    def __init__(self, candidates, classifier, *, seed=11, config=DEFAULT):
        self.candidates = candidates
        self.x = features(candidates)
        self.classifier = classifier.eval().requires_grad_(False)
        self.space = SearchSpace(candidates[0].kind, seed)
        self.config = config
        self.available = set(range(len(candidates)))

    def saliency(self, index):
        x = torch.tensor(self.x[index : index + 1], dtype=torch.float32, requires_grad=True)
        # The authors differentiate the failure-class logit, before softmax.
        gradient = torch.autograd.grad(self.classifier(x)[0, 1], x)[0][0].detach().numpy()
        weights = np.abs(gradient).astype(float)
        if weights.sum() == 0:
            encoded = int(self.space.generator.integers(len(weights)))
        else:
            encoded = int(self.space.generator.choice(len(weights), p=weights / weights.sum()))
        continuous = len(self.space.domain.bounds)
        if encoded < continuous:
            return encoded, float(gradient[encoded])
        offset = continuous
        for j, options in enumerate(self.space.domain.categories.values()):
            if encoded < offset + len(options):
                return continuous + j, None
            offset += len(options)
        raise ValueError("Saliency coordinate is outside the public scenario encoding")

    def probabilities(self, indices):
        with torch.no_grad():
            return self.classifier.probabilities(self.x[indices]).numpy()

    def project(self, scenario):
        available = sorted(self.available, key=lambda i: self.candidates[i].scenario_id)
        distance = ((self.x[available] - features([scenario])[0]) ** 2).sum(-1)
        return available[int(np.argmin(distance))]

    def propose(self):
        current = int(self.space.generator.choice(sorted(self.available)))
        score = float(self.probabilities([current])[0])
        stagnant, rounds = 0, 0
        for _ in range(self.config.max_neighborhoods):
            coordinate, direction = self.saliency(current)
            neighbors = [current]
            for _ in range(self.config.neighborhood_size - 1):
                proposal = self.space.mutate(
                    self.candidates[current], coordinate=coordinate, direction=direction
                )
                neighbors.append(self.project(proposal))
            values = self.probabilities(neighbors)
            best = int(np.argmax(values))
            improvement = float(values[best]) - score
            stagnant = stagnant + 1 if improvement <= self.config.stagnation_tolerance else 0
            current, score = neighbors[best], float(values[best])
            rounds += 1
            if stagnant >= self.config.max_stagnation:
                break
        return current, score, rounds


def run(candidates, oracle, *, classifier, initial_indices, budget=128, seed=11, config=DEFAULT):
    initial = list(initial_indices)
    validate_initial_indices(len(candidates), initial, budget)
    search = PoolSearch(candidates, classifier, seed=seed, config=config)
    queries, selected = [], []
    for step in range(budget):
        started = time.perf_counter()
        if step < len(initial):
            index = initial[step]
            probability, rounds = float(search.probabilities([index])[0]), 0
        else:
            index, probability, rounds = search.propose()
        selection_s = time.perf_counter() - started
        feedback = oracle.query(index, purpose="initial" if step < len(initial) else "target")
        if feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError("Indago feedback differs from its paid coordinate")
        search.available.remove(index)
        search.space.remember(candidates[index])
        selected.append(index)
        queries.append(
            {
                "query": step + 1,
                "index": index,
                "feedback": feedback.to_dict(),
                "prequery_probability": probability,
                "selection_s": selection_s,
                "adaptation_s": 0.0,
                "neighborhoods": rounds,
                "reason": "common_initialization"
                if step < len(initial)
                else "saliency_hill_climbing",
            }
        )
    return {
        "method": "indago",
        "queries": queries,
        "selected_indices": selected,
        "actual_queries": len(queries),
        "maximum_queries": budget,
        "termination": "budget_exhausted",
        "config": asdict(config),
        "feedback_used": "queried indices only; fixed historical classifier",
        "adaptation": "historical response training and nearest public candidate projection",
    }
