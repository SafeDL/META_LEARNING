"""Stop only when all working models predict sufficient retained discoveries."""

import torch

from methods.common.counts import count_cdf, expected_count


class StopController:
    def __init__(self, network, threshold):
        self.network = network.eval().requires_grad_(False)
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self, models, remaining, anchor, queries, evidence, budget):
        length = budget - len(queries)
        found = sum(q["risk"] > 0.5 for q in queries)
        features = self.network.features(
            models, remaining, anchor, queries, evidence, budget, [length]
        )
        parameters = self.network(features)[:, 0]
        probability = count_cdf(parameters, length, found // 9).cpu().numpy()
        mean = expected_count(parameters, length).cpu().numpy()
        return {
            "query": len(queries),
            "horizon": length,
            "found": found,
            "allowed_additional": found // 9,
            "threshold": self.threshold,
            "model_retention_probability": probability.tolist(),
            "model_remaining_mean": mean.tolist(),
            "count_parameters": parameters.cpu().tolist(),
            "retention_probability": float(probability.min()),
            "stop": bool(probability.min() >= self.threshold),
        }
