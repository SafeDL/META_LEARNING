"""Public candidate and paid feedback; trajectories belong to the evaluator."""

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    kind: str
    parameters: dict[str, float]
    categories: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.scenario_id or not self.kind:
            raise ValueError("A scenario needs an identifier and functional type")
        if not all(math.isfinite(v) for v in self.parameters.values()):
            raise ValueError("Scenario parameters must be finite")

    def coordinate(self):
        return (
            self.kind,
            tuple(sorted(self.parameters.items())),
            tuple(sorted(self.categories.items())),
        )


@dataclass(frozen=True)
class SafetyMeasurements:
    min_clearance_m: float
    min_ttc_s: float
    violations: tuple[str, ...] = ()
    impact_delta_v_mps: float | None = None
    driving_quality_score: float | None = None

    def __post_init__(self):
        if math.isnan(self.min_clearance_m) or math.isnan(self.min_ttc_s):
            raise ValueError("Safety measurements cannot contain NaN")
        if self.min_ttc_s < 0:
            raise ValueError("TTC must be nonnegative")
        if self.impact_delta_v_mps is not None and (
            not math.isfinite(self.impact_delta_v_mps) or self.impact_delta_v_mps < 0
        ):
            raise ValueError("Measured impact delta-V must be finite and nonnegative")
        if self.driving_quality_score is not None and (
            not math.isfinite(self.driving_quality_score) or self.driving_quality_score > 0
        ):
            raise ValueError("Driving quality is a finite nonpositive penalty score")


@dataclass(frozen=True)
class QueryFeedback:
    scenario_id: str
    collision: bool
    safety: SafetyMeasurements
    execution_s: float

    def __post_init__(self):
        if type(self.collision) is not bool:
            raise TypeError("Collision supervision must be a measured Boolean label")
        if not math.isfinite(self.execution_s) or self.execution_s < 0:
            raise ValueError("Execution cost must be finite and nonnegative")

    def to_dict(self):
        value = asdict(self)
        # JSON null represents an unbounded measured margin/TTC, never a missing label.
        for name in ("min_clearance_m", "min_ttc_s"):
            if math.isinf(value["safety"][name]):
                value["safety"][name] = None
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        safety = dict(value.pop("safety"))
        for name in ("min_clearance_m", "min_ttc_s"):
            if safety[name] is None:
                safety[name] = math.inf
        safety["violations"] = tuple(safety.get("violations", ()))
        return cls(safety=SafetyMeasurements(**safety), **value)
