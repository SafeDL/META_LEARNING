"""Explicit online GP fitting settings for development and registered experiments."""

import copy
import math
from dataclasses import dataclass

import torch

from .inference import TARGET_PARAMETERS, joint_evidence


@dataclass(frozen=True)
class FitSettings:
    steps: int
    learning_rate: float
    parameters: tuple[str, ...] = TARGET_PARAMETERS

    def __post_init__(self):
        if self.steps < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("GP fitting requires positive steps and a finite positive rate")
        if not self.parameters or len(set(self.parameters)) != len(self.parameters):
            raise ValueError("GP fitting requires distinct scalar parameters")

    def to_dict(self):
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "parameters": list(self.parameters),
        }


@dataclass(frozen=True)
class OnlineSettings:
    raw: FitSettings
    learned: FitSettings
    raw_initial_mean: float

    def __post_init__(self):
        if not math.isfinite(self.raw_initial_mean):
            raise ValueError("Raw GP initial mean must be finite")

    def to_dict(self):
        return {
            "raw": self.raw.to_dict(),
            "learned": self.learned.to_dict(),
            "raw_initial_mean": self.raw_initial_mean,
        }


def adapted_online(base, x, labels, margins, settings, *, encoded=None, implicit_evidence=False):
    """Reset at every paid feedback set and keep embeddings frozen."""
    memo = {}
    if not any(p.requires_grad for p in base.embedding.parameters()):
        memo[id(base.embedding)] = base.embedding
    if hasattr(base, "construction_specification"):
        memo[id(base.construction_specification)] = base.construction_specification
    model = copy.deepcopy(base, memo).double().eval()
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in settings.parameters)
        if parameter.requires_grad:
            if parameter.numel() != 1:
                raise ValueError("Only scalar GP parameters may be fitted online")
            parameters.append((name, parameter))
    if {n for n, _ in parameters} != set(settings.parameters):
        raise ValueError("Unknown online GP parameter")
    diagnostics = {"paid_observations": len(labels), "settings": settings.to_dict()}
    if len(labels):
        if encoded is None:
            encoded = model.features(x).detach()
        optimizer = torch.optim.Adam([p for _, p in parameters], lr=settings.learning_rate)

        def objective():
            return joint_evidence(model, x, labels, margins, encoded=encoded,
                                  implicit_evidence=implicit_evidence)

        with torch.enable_grad():
            for _ in range(settings.steps):
                optimizer.zero_grad(set_to_none=True)
                loss = objective()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite configured online GP objective")
                loss.backward()
                for group in model.adaptation_groups():
                    norm = torch.nn.utils.clip_grad_norm_(group, 5)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite configured online GP gradient")
                optimizer.step()
            final = objective()
            gradients = torch.autograd.grad(final, [p for _, p in parameters])
            gradient_max = max(float(g.detach().abs().max()) for g in gradients)
            if not torch.isfinite(final) or not math.isfinite(gradient_max):
                raise FloatingPointError("Nonfinite final online GP objective or gradient")
            diagnostics.update(objective=float(final.detach()), gradient_max=gradient_max)
    model.adaptation_diagnostics = diagnostics
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
