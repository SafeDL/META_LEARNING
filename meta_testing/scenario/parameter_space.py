"""Decode a policy's normalized hybrid action into executable parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from .option import AdversarialOption


@dataclass(frozen=True)
class NormalizedScenarioAction:
    candidate_index: int
    continuous: tuple[float, ...]
    option: AdversarialOption

    def validate(self, continuous_dim: int) -> None:
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        if len(self.continuous) != continuous_dim:
            raise ValueError(f"expected {continuous_dim} continuous controls")
        if not np.isfinite(self.continuous).all() or any(abs(value) > 1.0 for value in self.continuous):
            raise ValueError("normalized controls must be finite and lie in [-1, 1]")


@dataclass(frozen=True)
class ParameterSpace:
    parameter_space_id: str
    candidates: tuple[str, ...]
    bounds: Mapping[str, tuple[float, float]]
    options: tuple[AdversarialOption, ...] = tuple(AdversarialOption)

    def __post_init__(self) -> None:
        if not self.parameter_space_id or not self.candidates or not self.bounds:
            raise ValueError("parameter space requires an id, candidates, and bounds")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidate ids must be unique")
        for name, (lower, upper) in self.bounds.items():
            if not name or not np.isfinite([lower, upper]).all() or not lower < upper:
                raise ValueError(f"invalid bounds for {name!r}")

    @property
    def continuous_dim(self) -> int:
        return len(self.bounds)

    def decode(self, action: NormalizedScenarioAction) -> dict[str, float | str]:
        action.validate(self.continuous_dim)
        if action.candidate_index >= len(self.candidates):
            raise ValueError("candidate_index is outside the legal candidate set")
        if action.option not in self.options:
            raise ValueError("option is not enabled by this parameter space")
        result: dict[str, float | str] = {"route_or_conflict_candidate": self.candidates[action.candidate_index], "option": action.option.value}
        for (name, (lower, upper)), value in zip(self.bounds.items(), action.continuous):
            result[name] = float(lower + (float(value) + 1.0) * 0.5 * (upper - lower))
        return result

    def encode(self, values: Mapping[str, float | str]) -> NormalizedScenarioAction:
        candidate = str(values["route_or_conflict_candidate"])
        option = AdversarialOption(str(values["option"]))
        if candidate not in self.candidates:
            raise ValueError("unknown route_or_conflict_candidate")
        controls = []
        for name, (lower, upper) in self.bounds.items():
            value = float(values[name])
            if not lower <= value <= upper:
                raise ValueError(f"{name} is outside its legal bounds")
            controls.append(2.0 * (value - lower) / (upper - lower) - 1.0)
        return NormalizedScenarioAction(self.candidates.index(candidate), tuple(controls), option)
