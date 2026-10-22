"""Continuous proposals with the same frozen historical model and binary lookahead."""

import numpy as np
import torch

from benchmark.scenarios import features

from .prior import component_prior, posterior
from .selection import choose


CONFIG = {"proposal_count": 32, "global_fraction": 0.5, "horizon": 5}


class ContinuousDiscovery:
    def __init__(self, space, initial, encoder, history, *, target_weight=0.25, config=CONFIG):
        self.space = space
        self.scenes = list(initial)
        self.encoder = encoder
        self.history = history
        self.target_weight = target_weight
        self.config = config
        self.initial_count = len(initial)
        self.model = posterior(encoder, features(initial), history, target_weight=target_weight)
        self.pending = None

    def probability(self, scenario):
        index = next(i for i, s in enumerate(self.scenes) if s.scenario_id == scenario.scenario_id)
        return float(self.model.probabilities([index])[0])

    def propose(self, remaining):
        count = self.config["proposal_count"]
        global_count = round(count * self.config["global_fraction"])
        proposals = [self.space.random() for _ in range(global_count)]
        parent = self.scenes[int(np.argmax(self.model.probabilities()))]
        proposals += [self.space.mutate(parent) for _ in range(count - global_count)]
        old_count = len(self.scenes)
        joint = self.scenes + proposals
        with torch.no_grad():
            means, covariances, _ = component_prior(
                self.encoder, features(joint), self.history, target_weight=self.target_weight)
        means, covariances = means.cpu().numpy(), covariances.cpu().numpy()
        self.model.extend(means[:, old_count:], covariances[:, old_count:, :old_count],
                          covariances[:, old_count:, old_count:])
        self.scenes = joint
        available = np.arange(old_count, len(joint))
        horizon = min(remaining, self.config["horizon"], len(available))
        index, score = choose(self.model, available, [s.scenario_id for s in joint], horizon)
        self.pending = index
        return joint[index], float(self.model.probabilities([index])[0]), {
            "horizon": horizon, "acquisition": score, "proposals": count}

    def observe(self, scenario, feedback):
        index = (self.pending if self.pending is not None else
                 next(i for i, s in enumerate(self.scenes) if s.scenario_id == scenario.scenario_id))
        self.model.observe(index, feedback.collision)
        self.pending = None
        # During common initialization the remaining shared seeds must stay in the prior.
        if len(self.model.observed) >= self.initial_count:
            keep = sorted(self.model.observed)
            self.scenes = [self.scenes[i] for i in keep]
            self.model.retain(keep)
        return {"source_names": self.model.component_names, "source_weights": self.model.weights.tolist()}
