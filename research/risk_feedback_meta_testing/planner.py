"""One risk observation and batch continuation with a terminal-count constraint."""
import math

import torch

from .config import (EXPLOITATION_ROOTS, INTERIOR_QUADRATURE_POINTS,
                     UNCERTAINTY_ROOTS)
from .model import collision_probabilities, mixture_log_probabilities
from .session import MetaTestingSession


def continuation_gains(current_probability, future_probability, available,
                       remaining_budget):
    """Separate changed choices from changes in confidence on fixed choices."""
    prior = torch.topk(current_probability.masked_fill(~available, -torch.inf),
                       remaining_budget)
    future = torch.topk(
        future_probability.masked_fill(~available[None, :], -torch.inf),
        remaining_budget, dim=-1).values
    fixed = future_probability[:, prior.indices]
    time_weights = torch.arange(remaining_budget, 0, -1,
                                dtype=future.dtype, device=future.device)
    terminal_gain = (future.sum(-1) - fixed.sum(-1)).clamp_min(0)
    area_gain = ((future - fixed) * time_weights).sum(-1).clamp_min(0)
    return (prior.values.sum(), (prior.values * time_weights).sum(),
            terminal_gain, area_gain,
            future.sum(-1), (future * time_weights).sum(-1),
            fixed.sum(-1), (fixed * time_weights).sum(-1))


class BudgetLookaheadSession(MetaTestingSession):

    def __init__(self, *args, constrained=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrained = constrained

    def root_candidates(self, log_odds):
        remaining = int(self.remaining.sum())
        exploitation = torch.topk(log_odds.masked_fill(~self.remaining, -torch.inf),
                                  min(EXPLOITATION_ROOTS, remaining),
                                  largest=True).indices
        sensitivity = self.slope[self.family] / self.scale[self.family]
        eta = (self.logits + sensitivity[None, :]
               * (self.state.mean - self.state.baseline - self.center[self.family]))
        normal_scale = torch.sqrt(
            1 + math.pi / 8 * sensitivity.square() * self.state.variance)
        component_probability = torch.sigmoid(eta / normal_scale)
        weights = self.state.log_weights.exp()[:, None]
        probability = (weights * component_probability).sum(0)
        between = (weights * (component_probability - probability).square()).sum(0)
        derivative = (sensitivity[None, :] / normal_scale
                      * component_probability * (1 - component_probability))
        within = (weights * derivative.square() * self.state.variance).sum(0)
        uncertainty = (between + within).masked_fill(~self.remaining, -torch.inf)
        informative = torch.topk(uncertainty, min(UNCERTAINTY_ROOTS, remaining)).indices
        greedy = int(log_odds.masked_fill(~self.remaining, -torch.inf).argmax())
        others = torch.unique(torch.cat((exploitation, informative))).tolist()
        return [greedy] + [index for index in others if index != greedy]

    def root_scores(self, index, current_probability):
        budget = self.budget - self.count
        nodes, weights = self.state.predictive_nodes(index, INTERIOR_QUADRATURE_POINTS)
        posterior_weights, mean, variance = self.state.conditional_candidates(index, nodes)
        log_collision, log_safe = mixture_log_probabilities(
            self.logits, self.family, posterior_weights,
            mean - self.state.baseline[None, :, :], variance,
            self.slope, self.center, self.scale)
        probability = collision_probabilities(log_collision, log_safe)
        available = self.remaining.clone()
        available[index] = False
        prior_terminal, prior_area, terminal_gain, area_gain, raw_terminal, raw_area, \
            fixed_terminal, fixed_area = continuation_gains(
                current_probability, probability, available, budget - 1)
        terminal = current_probability[index] + prior_terminal + weights @ terminal_gain
        area = budget * current_probability[index] + prior_area + weights @ area_gain
        return {
            "terminal": float(terminal), "area": float(area),
            "terminal_choice_gain": float(weights @ terminal_gain),
            "area_choice_gain": float(weights @ area_gain),
            "raw_terminal": float(current_probability[index] + weights @ raw_terminal),
            "raw_area": float(budget * current_probability[index] + weights @ raw_area),
            "fixed_terminal_drift": float(weights @ fixed_terminal - prior_terminal),
            "fixed_area_drift": float(weights @ fixed_area - prior_area),
        }

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        if self.budget - self.count == 1:
            return super().next_index()
        log_collision, log_safe = self.log_probabilities()
        probability = collision_probabilities(log_collision, log_safe)
        roots = self.root_candidates(log_collision - log_safe)
        scores = {index: self.root_scores(index, probability) for index in roots}
        greedy = roots[0]
        selected = greedy
        for index in roots[1:]:
            terminal, area = scores[index]["terminal"], scores[index]["area"]
            if self.constrained and terminal < scores[greedy]["terminal"]:
                continue
            if area > scores[selected]["area"]:
                selected = index
        self.pending = selected
        self.records.append({
            "index": selected, "query_number": self.count + 1,
            "collision_probability": float(probability[selected]),
            "log_safe_probability": float(log_safe[selected]),
            "remaining_budget": self.budget - self.count,
            "greedy_reference_index": greedy,
            "predicted_terminal_count": scores[selected]["terminal"],
            "reference_terminal_count": scores[greedy]["terminal"],
            "predicted_cumulative_count": scores[selected]["area"],
            "reference_cumulative_count": scores[greedy]["area"],
            "terminal_choice_gain": scores[selected]["terminal_choice_gain"],
            "area_choice_gain": scores[selected]["area_choice_gain"],
            "fixed_choice_terminal_drift": scores[selected]["fixed_terminal_drift"],
            "fixed_choice_area_drift": scores[selected]["fixed_area_drift"],
            "constraint_enabled": self.constrained,
            "root_candidates": roots,
        })
        return self.pending
