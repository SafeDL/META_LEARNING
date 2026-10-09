"""Trainable failure head on the same frozen risk-response representation."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.behavior_response_testing.model import BehaviorResponseModel


PREDICTION_BATCH = 65536


def mixture_log_probabilities(logits, family, log_weights, residual_mean,
                              residual_variance, slope, center, scale):
    logits, log_weights, residual_mean, residual_variance, slope, center, scale = (
        value.to(dtype=torch.float64) for value in
        (logits, log_weights, residual_mean, residual_variance, slope, center, scale)
    )
    sensitivity = slope[family] / scale[family]
    eta = logits + sensitivity[None, :] * (residual_mean - center[family])
    denominator = torch.sqrt(
        1 + math.pi / 8 * sensitivity.square() * residual_variance)
    eta = eta / denominator
    log_collision = torch.logsumexp(
        log_weights[..., None] + F.logsigmoid(eta), -2)
    log_safe = torch.logsumexp(
        log_weights[..., None] + F.logsigmoid(-eta), -2)
    return log_collision.clamp_max(0), log_safe.clamp_max(0)


def collision_probabilities(log_collision, log_safe):
    """Use the stable side of the Bernoulli probability at both extremes."""
    return torch.where(log_safe < -math.log(2), -log_safe.expm1(),
                       log_collision.exp())


class FailureDecoder(nn.Module):
    """Both training modes have identical parameters and initialization."""

    def __init__(self, weight, bias, center, scale):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone())
        self.slope = nn.Parameter(bias.new_zeros(2))
        self.register_buffer("center", torch.as_tensor(center, dtype=bias.dtype,
                                                       device=bias.device))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=bias.dtype,
                                                      device=bias.device))
        if not torch.all(self.scale > 0):
            raise ValueError("Training residual scales must be positive")

    def logits(self, features, family):
        return ((features * self.weight[family][None, :, :]).sum(-1)
                + self.bias[family][None, :])

    def log_probabilities(self, features, family, log_weights, residual_mean,
                          residual_variance):
        return mixture_log_probabilities(
            self.logits(features, family), family, log_weights, residual_mean,
            residual_variance, self.slope, self.center, self.scale)


class FrozenResponseBackbone(nn.Module):

    def __init__(self, states):
        super().__init__()
        self.models = nn.ModuleList([BehaviorResponseModel() for _ in (0, 1)])
        for family, model in enumerate(self.models):
            model.load_state_dict(states[str(family)])
        self.requires_grad_(False)
        self.eval()

    def initial_head(self):
        return (torch.stack([model.layers[-1].weight[1]
                             for model in self.models]),
                torch.stack([model.layers[-1].bias[1]
                             for model in self.models]))

    @torch.no_grad()
    def features(self, x, behaviors):
        count = len(behaviors)
        risks = x.new_empty((count, len(x)))
        hidden = x.new_empty((count, len(x), 128))
        for family, model in enumerate(self.models):
            indices = torch.nonzero(x[:, 4] == family, as_tuple=True)[0]
            total = count * len(indices)
            for start in range(0, total, PREDICTION_BATCH):
                flat = torch.arange(start, min(start + PREDICTION_BATCH, total),
                                    device=x.device)
                hypothesis = flat // len(indices)
                scene = indices[flat % len(indices)]
                inputs = torch.cat((x[scene, :4], behaviors[hypothesis]), 1)
                values = model.layers[:-1](inputs)
                hidden[hypothesis, scene] = values
                risks[hypothesis, scene] = model.layers[-1](values)[:, 0].sigmoid()
        return risks, hidden

    @torch.no_grad()
    def tables(self, x, behaviors, decoder):
        """Inference avoids retaining the full particle-by-scene feature tensor."""
        risks = x.new_empty((len(behaviors), len(x)))
        logits = torch.empty_like(risks)
        for family, model in enumerate(self.models):
            indices = torch.nonzero(x[:, 4] == family, as_tuple=True)[0]
            total = len(behaviors) * len(indices)
            for start in range(0, total, PREDICTION_BATCH):
                flat = torch.arange(start, min(start + PREDICTION_BATCH, total),
                                    device=x.device)
                hypothesis = flat // len(indices)
                scene = indices[flat % len(indices)]
                inputs = torch.cat((x[scene, :4], behaviors[hypothesis]), 1)
                hidden = model.layers[:-1](inputs)
                risks[hypothesis, scene] = model.layers[-1](hidden)[:, 0].sigmoid()
                logits[hypothesis, scene] = (
                    (hidden * decoder.weight[family]).sum(-1) + decoder.bias[family])
        return risks, logits
