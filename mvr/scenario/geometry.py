"""Concrete, reproducible MetaDrive geometry definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GeometrySpec:
    geometry_id: str
    functional_scenario: str
    map_code: str
    seed: int
    split: str
    random_lane_width: bool = True
    random_lane_num: bool = True
    horizon: int = 180
    lane_num: int | None = None
    lane_width: float | None = None

    def validate(self) -> None:
        if not self.geometry_id or not self.functional_scenario or not self.map_code:
            raise ValueError("geometry requires id, functional scenario, and map code")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("geometry split must be train, validation, or test")
        if not isinstance(self.seed, int) or self.horizon < 1:
            raise ValueError("geometry seed and horizon must be valid")
        if self.lane_num is not None and self.lane_num < 1:
            raise ValueError("geometry lane_num must be positive")
        if self.lane_width is not None and self.lane_width <= 0.0:
            raise ValueError("geometry lane_width must be positive")

    def env_overrides(self) -> dict[str, Any]:
        self.validate()
        result = {
            "map": self.map_code,
            "start_seed": self.seed,
            "num_scenarios": 1,
            "horizon": self.horizon,
            "random_lane_width": self.random_lane_width,
            "random_lane_num": self.random_lane_num,
        }
        if self.lane_num is not None or self.lane_width is not None:
            result["map_config"] = {
                key: value
                for key, value in {
                    "lane_num": self.lane_num,
                    "lane_width": self.lane_width,
                }.items()
                if value is not None
            }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeometrySpec":
        geometry = cls(**dict(value))
        geometry.validate()
        return geometry
