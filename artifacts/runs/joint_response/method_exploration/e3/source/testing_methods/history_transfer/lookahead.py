"""One-step joint-response fantasies followed by a remaining-budget batch."""

import math

import numpy as np
import torch

from .inference import (
    JointPosterior,
    JointPrior,
    adapted_model,
    mixture_weights,
    predict_components,
    update_weights,
)
from .model import MARGIN_NOISE


def rank(indices, values, candidate_ids):
    """Sort descending; resolve each 1e-12 group by scenario ID."""
    values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    ordered = sorted(indices, key=lambda i: -float(values[i]))
    result = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] >= values[ordered[start]] - 1e-12:
            end += 1
        result.extend(sorted(ordered[start:end], key=lambda i: candidate_ids[i]))
        start = end
    return result


def shortlist(remaining, components, weights, candidate_ids, count):
    probability = (weights[:, None] * components).sum(0)
    disagreement = (weights[:, None] * (components - probability) ** 2).sum(0)
    exploitation = rank(remaining, probability, candidate_ids)
    exploration = rank(remaining, disagreement, candidate_ids)
    selected = exploitation[: max(1, count // 2)]
    for index in exploration + exploitation:
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return selected


def sample_responses(means, blocks, weights, uniforms):
    """Sample one mixture component, its (f, r), then Bernoulli-logistic y.

    RSS fantasies are Gaussian model responses. They are not physical RSS
    measurements and may fall below the physical margin bound of -1.
    """
    component = torch.searchsorted(weights.cumsum(0), uniforms[:, 0].contiguous())
    covariance = blocks[component].clone()
    covariance[:, 1, 1] += MARGIN_NOISE**2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-9:
        raise ValueError("The joint fantasy covariance is not positive semidefinite")
    root = eigenvectors * eigenvalues.clamp_min(0).sqrt()[:, None, :]
    normal = math.sqrt(2) * torch.erfinv(2 * uniforms[:, 1:3] - 1)
    values = means[component] + torch.einsum("nij,nj->ni", root, normal)
    labels = uniforms[:, 3] < values[:, 0].sigmoid()
    return labels.cpu().tolist(), values[:, 1].cpu().tolist()


class JointLookahead:
    """Share frozen inputs across fantasies; keep every branch outside the oracle."""

    def __init__(
        self,
        base,
        physical_base,
        x,
        sources,
        reference,
        encoded_target,
        encoded_physical,
        candidate_ids,
        *,
        shortlist_count,
        fantasy_count,
        seed,
        batch_size,
    ):
        self.base, self.physical_base = base, physical_base
        self.x, self.sources, self.reference = x, sources, reference
        self.encoded_target, self.encoded_physical = encoded_target, encoded_physical
        self.candidate_ids = candidate_ids
        self.shortlist_count, self.fantasy_count = shortlist_count, fantasy_count
        self.seed, self.batch_size = seed, batch_size

    def point_posteriors(self, model, physical, paid, labels, margins, indices):
        """Use exactly the point blocks, source approximation and Laplace update."""
        priors = [JointPrior(model, self.x, self.reference, s) for s in self.sources]
        priors.extend(
            (
                JointPrior(model, self.x, self.reference),
                JointPrior(physical, self.x, self.reference),
            )
        )
        means, blocks = [], []
        for prior in priors:
            encoded = self.encoded_physical if prior.model is physical else self.encoded_target
            observed = prior.model.kernel_from_features(encoded[paid])
            state = JointPosterior(
                prior.mean(paid),
                prior.covariance(paid, paid, base_covariance=observed),
                labels,
                margins,
            )
            cross = prior.model.kernel_from_features(encoded[indices], encoded[paid])
            mean, block = state.predict(
                prior.mean(indices),
                prior.covariance(indices, paid, base_covariance=cross),
                prior.blocks(indices),
            )
            means.append(mean)
            blocks.append(block)
        return torch.stack(means), torch.stack(blocks)

    def fantasy_prediction(self, scores, paid, labels, margins, index, collision, rss, prequery):
        """Apply the real update order from the same fixed initialization."""
        next_scores, _ = update_weights(scores, prequery[:-1], collision)
        next_paid = [*paid, index]
        next_labels, next_margins = [*labels, collision], [*margins, rss]
        model = adapted_model(
            self.base,
            self.x[next_paid],
            next_labels,
            next_margins,
            learning_rate=0.003,
            encoded=self.encoded_target[next_paid],
        )
        physical = adapted_model(
            self.physical_base,
            self.x[next_paid],
            next_labels,
            next_margins,
            learning_rate=0.01,
            encoded=self.encoded_physical[next_paid],
        )
        with torch.no_grad():
            components, _ = predict_components(
                model,
                physical,
                self.x,
                self.sources,
                self.reference,
                next_paid,
                next_labels,
                next_margins,
                batch_size=self.batch_size,
                encoded_target=self.encoded_target,
                encoded_physical=self.encoded_physical,
            )
        return (mixture_weights(next_scores)[:, None] * components).sum(0)

    def choose(self, model, physical, scores, paid, labels, margins, components, remaining, budget):
        weights = mixture_weights(scores)
        probability = (weights[:, None] * components).sum(0)
        if budget == 1:
            index = rank(remaining, probability, self.candidate_ids)[0]
            return index, {"fantasy_branches": 0, "remaining_budget": 1}
        indices = shortlist(
            remaining, components, weights, self.candidate_ids, self.shortlist_count
        )
        with torch.no_grad():
            means, blocks = self.point_posteriors(model, physical, paid, labels, margins, indices)
        engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=self.seed + len(paid))
        uniforms = engine.draw(self.fantasy_count, dtype=torch.float64).to(model.mean.device)
        uniforms = uniforms.clamp(
            torch.finfo(uniforms.dtype).eps, 1 - torch.finfo(uniforms.dtype).eps
        )
        records, utilities = [], probability.new_full(probability.shape, -torch.inf)
        for position, index in enumerate(indices):
            fake_labels, fake_margins = sample_responses(
                means[:, position], blocks[:, position], weights, uniforms
            )
            future_indices = sorted(set(remaining) - {index})
            future_utilities = []
            for collision, rss in zip(fake_labels, fake_margins):
                after = self.fantasy_prediction(
                    scores, paid, labels, margins, index, collision, rss, components[:, index]
                )
                top = torch.topk(after[future_indices], k=budget - 1).values
                future_utilities.append(float(top.sum()))
            utility = float(probability[index]) + float(np.mean(future_utilities))
            utilities[index] = utility
            rss_sd = (blocks[:, position, 1, 1] + MARGIN_NOISE**2).clamp_min(1e-12).sqrt()
            tail = 0.5 * (1 + torch.erf((-1 - means[:, position, 1]) / (math.sqrt(2) * rss_sd)))
            records.append(
                {
                    "index": index,
                    "probability": float(probability[index]),
                    "utility": utility,
                    "future_utility": future_utilities,
                    "rss_below_physical_bound_probability": float(weights @ tail),
                    "fantasies_below_physical_bound": sum(r < -1 for r in fake_margins),
                }
            )
        index = rank(indices, utilities, self.candidate_ids)[0]
        return index, {
            "remaining_budget": budget,
            "fantasy_branches": len(indices) * self.fantasy_count,
            "shortlist": records,
        }
