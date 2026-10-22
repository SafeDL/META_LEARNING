"""Continuous proposals and mixed categorical coordinates, without SUT outcomes."""

import numpy as np

from .schema import Scenario
from .scenarios import DOMAINS, difficult, features, validate


class SearchSpace:
    def __init__(self, kind, seed, *, ordinary_only=False):
        self.kind = kind
        self.domain = DOMAINS[kind]
        self.generator = np.random.default_rng(seed)
        self.seed = seed
        self.counter = 0
        self.ordinary_only = ordinary_only
        self.seen = set()
        self.dimension = len(self.domain.bounds) + len(self.domain.categories)

    def decode(self, vector):
        vector = np.clip(np.asarray(vector), 0, 1)
        parameters = {name: float(lo + vector[i] * (hi - lo))
                      for i, (name, (lo, hi)) in enumerate(self.domain.bounds.items())}
        categories = {name: choices[min(int(vector[len(parameters) + i] * len(choices)), len(choices) - 1)]
                      for i, (name, choices) in enumerate(self.domain.categories.items())}
        self.counter += 1
        return Scenario(f"{self.kind}:continuous:{self.seed}:{self.counter:09d}",
                        self.kind, parameters, categories)

    def vector(self, scenario):
        output = [(scenario.parameters[name] - lo) / (hi - lo)
                  for name, (lo, hi) in self.domain.bounds.items()]
        output += [(choices.index(scenario.categories[name]) + 0.5) / len(choices)
                   for name, choices in self.domain.categories.items()]
        return np.asarray(output)

    def valid(self, scenario, *, unqueried=True):
        try:
            validate(scenario)
        except ValueError:
            return False
        if self.ordinary_only and difficult(scenario):
            return False
        return not unqueried or scenario.coordinate() not in self.seen

    def remember(self, scenario):
        if scenario.coordinate() in self.seen:
            raise ValueError("Search algorithms must explicitly meter repeated coordinates")
        self.seen.add(scenario.coordinate())

    def random(self):
        while True:
            scenario = self.decode(self.generator.random(self.dimension))
            if self.valid(scenario):
                return scenario

    def mutate(self, parent, *, coordinate=None, direction=None, step=(0.01, 0.1)):
        while True:
            vector = self.vector(parent)
            selected = (int(self.generator.integers(self.dimension))
                        if coordinate is None else coordinate)
            if selected >= len(self.domain.bounds):
                name = list(self.domain.categories)[selected - len(self.domain.bounds)]
                choices = self.domain.categories[name]
                other = [i for i, value in enumerate(choices) if value != parent.categories[name]]
                vector[selected] = (int(self.generator.choice(other)) + 0.5) / len(choices)
            else:
                sign = int(self.generator.choice([-1, 1])) if direction is None or direction == 0 else np.sign(direction)
                vector[selected] += sign * self.generator.uniform(*step)
            scenario = self.decode(vector)
            if self.valid(scenario) and scenario.coordinate() != parent.coordinate():
                return scenario
            # At a physical boundary try the opposite direction, rather than returning an unchanged point.
            direction = None
            coordinate = None

    def feature_matrix(self, scenarios):
        return features(scenarios)


def initial_scenarios(kind, seed, count=8, *, ordinary_only=False):
    space = SearchSpace(kind, seed + 910003, ordinary_only=ordinary_only)
    output = []
    for _ in range(count):
        point = space.random()
        output.append(point)
        space.remember(point)
    return output
