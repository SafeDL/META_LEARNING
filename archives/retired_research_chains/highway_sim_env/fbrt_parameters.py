"""Static parameter metadata shared by scenario definitions and offline analysis."""

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
