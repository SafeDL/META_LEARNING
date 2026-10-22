"""Indago's random-start, saliency-guided hill climbing under shared feedback."""

import numpy as np
import torch


CONFIG = {"neighborhood_size": 50, "max_neighborhoods": 100}


class Indago:
    def __init__(self, space, classifier, *, config=CONFIG):
        self.space = space
        self.classifier = classifier.eval().requires_grad_(False)
        self.config = config

    def logits(self, scenes):
        with torch.no_grad():
            return self.classifier(self.space.feature_matrix(scenes)).numpy()

    def saliency(self, scene):
        x = torch.tensor(self.space.feature_matrix([scene]), dtype=torch.float32, requires_grad=True)
        gradient = torch.autograd.grad(self.classifier(x).sum(), x)[0][0].detach().numpy()
        continuous = len(self.space.domain.bounds)
        relevance = list(np.abs(gradient[:continuous]))
        offset = continuous
        for options in self.space.domain.categories.values():
            relevance.append(float(np.abs(gradient[offset:offset + len(options)]).max()))
            offset += len(options)
        coordinate = int(np.argmax(relevance))
        direction = float(gradient[coordinate]) if coordinate < continuous else None
        return coordinate, direction

    def propose(self, remaining):
        current = self.space.random()
        score = self.logits([current])[0]
        rounds = 0
        for _ in range(self.config["max_neighborhoods"]):
            coordinate, direction = self.saliency(current)
            neighbors = [self.space.mutate(current, coordinate=coordinate, direction=direction)
                         for _ in range(self.config["neighborhood_size"])]
            values = self.logits(neighbors)
            best = int(np.argmax(values))
            rounds += 1
            if values[best] <= score + 1e-10:
                break
            current, score = neighbors[best], values[best]
        probability = float(torch.sigmoid(torch.tensor(float(score))))
        return current, probability, {"search": "saliency_hill_climbing", "neighborhoods": rounds,
                                      "neighborhood_size": self.config["neighborhood_size"]}

    def observe(self, scenario, feedback):
        # The published classifier is fixed during failure search.
        return {}
