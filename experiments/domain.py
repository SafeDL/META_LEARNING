"""Explicit free and fixed scenario parameters, independent of the execution backend."""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Domain:
    bounds: dict[str, tuple[float, float]]
    quantization: dict[str, float]
    categories: dict[str, tuple[str, ...]]
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    fixed_categories: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self, "bounds", {k: tuple(map(float, v)) for k, v in self.bounds.items()}
        )
        object.__setattr__(
            self, "quantization", {k: float(v) for k, v in self.quantization.items()}
        )
        object.__setattr__(self, "categories", {k: tuple(v) for k, v in self.categories.items()})
        object.__setattr__(
            self, "fixed_parameters", {k: float(v) for k, v in self.fixed_parameters.items()}
        )
        object.__setattr__(self, "fixed_categories", dict(self.fixed_categories))
        if not self.bounds and not self.categories:
            raise ValueError("A candidate domain needs at least one free parameter")
        if self.bounds.keys() != self.quantization.keys():
            raise ValueError("Each continuous parameter needs a registered mutation step")
        names = [*self.bounds, *self.categories, *self.fixed_parameters, *self.fixed_categories]
        if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("Free and fixed parameter names must be distinct nonempty strings")
        for name, (lower, upper) in self.bounds.items():
            step = self.quantization[name]
            if not all(math.isfinite(v) for v in (lower, upper, step)) or not (
                lower < upper and 0 < step <= upper - lower
            ):
                raise ValueError("Continuous bounds and mutation steps must be finite and positive")
        if not all(math.isfinite(v) for v in self.fixed_parameters.values()):
            raise ValueError("Fixed numeric parameters must be finite")
        for options in self.categories.values():
            if (
                not options
                or len(set(options)) != len(options)
                or any(not isinstance(v, str) or not v for v in options)
            ):
                raise ValueError("Category options must be distinct nonempty strings")
        if any(not isinstance(v, str) or not v for v in self.fixed_categories.values()):
            raise ValueError("Fixed categories must be nonempty strings")

    @property
    def dimension(self):
        return len(self.bounds) + sum(len(v) for v in self.categories.values())

    def to_dict(self):
        # Lists bind feature order even when archive fingerprints sort JSON object keys.
        return {
            "bounds": [[name, *limits] for name, limits in self.bounds.items()],
            "quantization": dict(self.quantization),
            "categories": [[name, list(values)] for name, values in self.categories.items()],
            "fixed_parameters": dict(self.fixed_parameters),
            "fixed_categories": dict(self.fixed_categories),
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {
            "bounds",
            "quantization",
            "categories",
            "fixed_parameters",
            "fixed_categories",
        }:
            raise ValueError("The domain record must specify all free and fixed fields")
        bounds = {name: (lower, upper) for name, lower, upper in value["bounds"]}
        categories = {name: tuple(options) for name, options in value["categories"]}
        if len(bounds) != len(value["bounds"]) or len(categories) != len(value["categories"]):
            raise ValueError("Serialized feature names must be unique")
        return cls(
            bounds,
            value["quantization"],
            categories,
            value["fixed_parameters"],
            value["fixed_categories"],
        )

    def validate(self, scenario):
        """Check parameter coordinates; physical feasibility belongs to the executor."""
        if (
            scenario.parameters.keys() != self.bounds.keys()
            or scenario.categories.keys() != self.categories.keys()
        ):
            raise ValueError("Candidate fields must contain exactly the free domain parameters")
        for name, (lower, upper) in self.bounds.items():
            if not lower <= scenario.parameters[name] <= upper:
                raise ValueError(f"Parameter outside its registered range: {name}")
        for name, options in self.categories.items():
            if scenario.categories[name] not in options:
                raise ValueError(f"Category outside its registered options: {name}")

    def features(self, scenarios):
        if not scenarios or len({s.kind for s in scenarios}) != 1:
            raise ValueError("Features require one nonempty functional scenario task")
        rows = []
        for scene in scenarios:
            self.validate(scene)
            row = [(scene.parameters[n] - lo) / (hi - lo) for n, (lo, hi) in self.bounds.items()]
            row.extend(
                float(scene.categories[n] == v)
                for n, options in self.categories.items()
                for v in options
            )
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def parameter_distances(self, scenes, others):
        if not scenes or not others or len({s.kind for s in (*scenes, *others)}) != 1:
            raise ValueError("Parameter distances require one nonempty functional scenario task")
        for scene in (*scenes, *others):
            self.validate(scene)
        distances = np.zeros((len(scenes), len(others)), dtype=float)
        for name, (lower, upper) in self.bounds.items():
            left = np.array([s.parameters[name] for s in scenes])
            right = np.array([s.parameters[name] for s in others])
            distances += np.abs(left[:, None] - right) / (upper - lower)
        for name in self.categories:
            left = np.array([s.categories[name] for s in scenes])
            right = np.array([s.categories[name] for s in others])
            distances += left[:, None] != right
        return distances / (len(self.bounds) + len(self.categories))

    def execution_values(self, scenario):
        """Merge fixed primitives; derived geometry is computed by the executor."""
        self.validate(scenario)
        return (
            {**self.fixed_parameters, **scenario.parameters},
            {**self.fixed_categories, **scenario.categories},
        )
