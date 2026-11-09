"""Public candidates and measured responses disclosed after a paid execution."""

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
    rss_margin: float | None = None

    def __post_init__(self):
        if self.rss_margin is not None and (
            not math.isfinite(self.rss_margin) or self.rss_margin < -1 - 1e-12
        ):
            raise ValueError("RSS must be a finite normalized margin >= -1")
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
    rss_contract: dict | None = None
    trajectory: list[dict] | None = None

    def __post_init__(self):
        if type(self.collision) is not bool:
            raise TypeError("Collision supervision must be a measured Boolean label")
        if not math.isfinite(self.execution_s) or self.execution_s < 0:
            raise ValueError("Execution cost must be finite and nonnegative")
        if self.trajectory is not None:
            times = [frame["time_s"] for frame in self.trajectory]
            if (
                not times
                or any(not math.isfinite(t) or t < 0 for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))
            ):
                raise ValueError("Paid trajectory must have increasing measured timestamps")
            if any(bool(frame["collision"]) for frame in self.trajectory) != self.collision:
                raise ValueError("Paid trajectory and collision feedback disagree")

    def to_dict(self):
        value = asdict(self)
        # Preserve the serialized shape of recorded collision-only feedback.
        if value["rss_contract"] is None:
            value.pop("rss_contract")
        if value["trajectory"] is None:
            value.pop("trajectory")
        if value["safety"]["rss_margin"] is None:
            value["safety"].pop("rss_margin")
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
