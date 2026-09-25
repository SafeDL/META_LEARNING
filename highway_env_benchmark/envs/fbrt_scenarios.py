"""Physical parameters for four functional highway regression scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass


BOUNDS = {
    "fbrt_cutin": ((8.0, 50.0), (0.6, 3.0)),
    "fbrt_lead_emergency_brake": ((5.0, 120.0), (3.0, 8.0)),
    "fbrt_stop_hold_go": ((4.0, 35.0), (2.0, 5.0)),
    "fbrt_cutout_static": ((8.0, 35.0), (1.5, 4.0)),
}

ACTIVE_PARAMETERS = {
    "fbrt_cutin": ("initial_clearance_m", "lane_change_duration_s"),
    "fbrt_lead_emergency_brake": ("initial_clearance_m", "lead_deceleration_mps2"),
    "fbrt_stop_hold_go": ("initial_clearance_m", "lead_deceleration_mps2"),
    "fbrt_cutout_static": ("initial_clearance_m", "static_target_ttc_s"),
}


@dataclass(frozen=True)
class FBRTScenario:
    scenario_id: str
    template_id: str
    initial_clearance_m: float
    lane_change_duration_s: float | None = None
    lead_deceleration_mps2: float | None = None
    static_target_ttc_s: float | None = None

    @property
    def lane_count(self) -> int:
        return 1 if self.template_id == "fbrt_lead_emergency_brake" else 2

    @property
    def ego_speed_mps(self) -> float:
        return {"fbrt_cutin": 25.0, "fbrt_lead_emergency_brake": 20.0,
                "fbrt_stop_hold_go": 15.0, "fbrt_cutout_static": 20.0}[self.template_id]

    @property
    def lead_speed_mps(self) -> float:
        return 20.0 if self.template_id == "fbrt_cutin" else self.ego_speed_mps

    @property
    def duration_s(self) -> float:
        return {"fbrt_cutin": 9.0, "fbrt_lead_emergency_brake": 12.0,
                "fbrt_stop_hold_go": 24.0, "fbrt_cutout_static": 10.0}[self.template_id]

    def active_values(self) -> tuple[float, float]:
        names = ACTIVE_PARAMETERS[self.template_id]
        return tuple(float(getattr(self, name)) for name in names)

    def as_record(self) -> dict:
        return {**asdict(self), "lane_count": self.lane_count,
                "ego_speed_mps": self.ego_speed_mps,
                "lead_speed_mps": self.lead_speed_mps}
