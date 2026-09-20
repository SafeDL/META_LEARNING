"""Finite, time-aware sparse keys for the highway DenseRL replication."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StateEncoder:
    representation: str = "full"
    position_bin: float = 2.0
    speed_bin: float = 1.0
    lateral_bin: float = 1.0
    time_bin: float = 0.2
    heading_bin: float = 0.1
    clip_longitudinal: float = 80.0
    clip_speed: float = 40.0
    clip_heading: float = 3.2

    def encode(self, observation: dict[str, float]) -> tuple[int, ...]:
        def quantize(value: float, width: float, limit: float) -> int:
            return int(np.rint(np.clip(value, -limit, limit) / width))

        if self.representation == "passing":
            return (
                quantize(observation["npc_speed"], self.speed_bin, self.clip_speed),
                quantize(observation["leading_gap"], self.position_bin,
                         self.clip_longitudinal),
                quantize(observation["leading_relative_speed"], self.speed_bin,
                         self.clip_speed),
                quantize(observation["longitudinal_gap"], self.position_bin,
                         self.clip_longitudinal),
                quantize(observation["relative_speed"], self.speed_bin,
                         self.clip_speed),
                int(np.rint(observation["time"] / self.time_bin)),
            )
        if self.representation != "full":
            raise ValueError(f"unknown state representation: {self.representation}")
        return (
            quantize(observation["longitudinal_gap"], self.position_bin, self.clip_longitudinal),
            quantize(observation["lateral_gap"], self.lateral_bin, 12.0),
            quantize(observation["relative_speed"], self.speed_bin, self.clip_speed),
            quantize(observation["ego_speed"], self.speed_bin, self.clip_speed),
            quantize(observation["ego_heading"], self.heading_bin, self.clip_heading),
            quantize(observation["npc_heading"], self.heading_bin, self.clip_heading),
            int(np.rint(observation["npc_target_lane"])),
            int(np.rint(observation["schedule_phase"])),
            int(np.rint(observation["time"] / self.time_bin)),
            int(observation["mode_code"]),
        )
