"""Input-only feature adapters for DETOUR-Scenario-H and road unit checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replications.detour_highway_env.detour.contracts import ScenarioSpec

MODE_VOCABULARY = (
    "fast_intrusion",
    "cutin_braking",
    "lead_braking",
    "stop_and_go",
    "slow_lead_following",
)
PASSING_MODE = "passing_cutin"
GAP_RANGE, RELATIVE_SPEED_RANGE = (5.0, 40.0), (-8.0, 2.0)


@dataclass(frozen=True)
class RoadCurvatureFeatures:
    """Compact curvature description used only for the original-road unit check."""
    initial_heading: float
    lengths: np.ndarray
    curvatures: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray) -> "RoadCurvatureFeatures":
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError("points must have shape (n>=3, 2)")
        deltas, lengths = np.diff(values, axis=0), np.linalg.norm(np.diff(values, axis=0), axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("consecutive road points must be distinct")
        headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        # The first chord establishes the initial tangent.  Each subsequent
        # value is the signed heading change per metre for one arc segment.
        curvature = np.zeros_like(lengths)
        curvature[1:] = np.diff(headings) / lengths[1:]
        return cls(float(headings[0]), lengths, curvature)

    def compressed(self, max_segments: int) -> "RoadCurvatureFeatures":
        """Greedily merge the least shape-changing adjacent pair.

        This is the road-only counterpart of the representation in DETOUR:
        neighbouring segments are replaced by their length-weighted curvature.
        It is deliberately not used for the straight-road Cut-in experiment.
        """
        if max_segments < 1:
            raise ValueError("max_segments must be positive")
        lengths, curvatures = self.lengths.astype(float).tolist(), self.curvatures.astype(
            float).tolist()
        while len(lengths) > max_segments:
            losses = [
                abs(curvatures[index] - curvatures[index + 1]) *
                (lengths[index] + lengths[index + 1]) for index in range(len(lengths) - 1)
            ]
            index = int(np.argmin(losses))
            length = lengths[index] + lengths[index + 1]
            curvature = (lengths[index] * curvatures[index] +
                         lengths[index + 1] * curvatures[index + 1]) / length
            lengths[index:index + 2] = [length]
            curvatures[index:index + 2] = [curvature]
        return RoadCurvatureFeatures(self.initial_heading, np.asarray(lengths),
                                     np.asarray(curvatures))

    def reconstruct(self, start: np.ndarray = np.zeros(2)) -> np.ndarray:
        point, heading, points = np.asarray(start, dtype=float).copy(), self.initial_heading, []
        points.append(point.copy())
        for length, curvature in zip(self.lengths, self.curvatures, strict=True):
            turn = float(curvature * length)
            if abs(curvature) < 1e-12:
                delta = float(length) * np.array([np.cos(heading), np.sin(heading)])
            else:
                delta = np.array([
                    (np.sin(heading + turn) - np.sin(heading)) / curvature,
                    (-np.cos(heading + turn) + np.cos(heading)) / curvature,
                ])
            point = point + delta
            heading += turn
            points.append(point.copy())
        return np.asarray(points)


def encode_scenarios(scenarios: tuple[ScenarioSpec, ...] | list[ScenarioSpec]) -> np.ndarray:
    """Encode input-only continuous fields plus a non-ordinal mode one-hot block."""
    vocabulary = (
        MODE_VOCABULARY + (PASSING_MODE,)
        if any(scenario.mode == PASSING_MODE for scenario in scenarios)
        else MODE_VOCABULARY
    )
    encoded = []
    include_controls = any(scenario.timing is not None for scenario in scenarios)
    for scenario in scenarios:
        if scenario.mode not in vocabulary:
            raise ValueError(f"unsupported Cut-in mode: {scenario.mode}")
        gap = (scenario.initial_gap - GAP_RANGE[0]) / (GAP_RANGE[1] - GAP_RANGE[0])
        speed = (scenario.relative_speed - RELATIVE_SPEED_RANGE[0]) / (RELATIVE_SPEED_RANGE[1] -
                                                                       RELATIVE_SPEED_RANGE[0])
        one_hot = np.zeros(len(vocabulary), dtype=float)
        one_hot[vocabulary.index(scenario.mode)] = 1.0 / np.sqrt(2.0)
        continuous = [gap, speed]
        if include_controls:
            continuous.extend((float(scenario.timing), float(scenario.intensity)))
        encoded.append(np.concatenate((continuous, one_hot)))
    return np.asarray(encoded, dtype=float)
