"""Predeclared, same-controller IDM software releases for bidirectional testing.

These are engineering release intentions, not profiles selected for test-bank
performance.  V1 changes the normal following calibration; V2 inherits every
V1 field and adds only the short-TTC safeguard.
"""

from __future__ import annotations

from dataclasses import replace

from .idm_profiles import SUTProfile


NL_V0 = SUTProfile("nl_v0", "IDM", time_wanted=1.5, desired_gap=5.0,
                   comfort_acceleration=2.5, max_brake=5.0)
NL_V1 = replace(NL_V0, name="nl_v1", time_wanted=1.2, desired_gap=4.0,
                comfort_acceleration=2.8, max_brake=5.5)
NL_V2 = replace(NL_V1, name="nl_v2", emergency_ttc=3.0,
                emergency_brake_gain=8.0, emergency_max_brake=11.0)

NL_RELEASES = {profile.name: profile for profile in (NL_V0, NL_V1, NL_V2)}
NL_PARENTS = {"nl_v0": None, "nl_v1": "nl_v0", "nl_v2": "nl_v1"}
NL_PURPOSES = {
    "nl_v0": "Reference IDM baseline",
    "nl_v1": "Efficiency and normal-response calibration",
    "nl_v2": "Short-TTC predictive-brake safeguard",
}

# Candidate calibrations for a *separate* development study. They do not
# change the frozen nl_v0/v1/v2 release chain or either of its measured banks.
# Selection is by measured progress with a predeclared collision constraint,
# never by bidirectional selector performance.
NL_EFFICIENCY_CANDIDATES = {
    "nl_eff_c1": replace(NL_V0, name="nl_eff_c1", time_wanted=1.0,
                         desired_gap=3.5, comfort_acceleration=2.8),
    "nl_eff_c2": replace(NL_V0, name="nl_eff_c2", time_wanted=0.8,
                         desired_gap=3.0, comfort_acceleration=3.0),
    "nl_eff_c3": replace(NL_V0, name="nl_eff_c3", time_wanted=0.6,
                         desired_gap=2.5, comfort_acceleration=3.2),
}

# Second chain: V1 is the calibration study's selected c2, copied with a
# release identity. V2 keeps all its normal-following fields and adds only the
# same short-TTC safeguard family as the first chain.
NL2_V1 = replace(NL_EFFICIENCY_CANDIDATES["nl_eff_c2"], name="nl2_v1")
NL2_V2 = replace(NL2_V1, name="nl2_v2", emergency_ttc=3.0,
                 emergency_brake_gain=8.0, emergency_max_brake=11.0)
NL2_RELEASES = {profile.name: profile for profile in (NL2_V1, NL2_V2)}
NL2_PARENTS = {"nl2_v1": "nl_v0", "nl2_v2": "nl2_v1"}
NL2_PURPOSES = {
    "nl2_v1": "Efficiency calibration selected from independent progress pilot",
    "nl2_v2": "Short-TTC predictive-brake safeguard inherited from nl2_v1",
}

# New development chain after checking the longitudinal geometry contract.
# highway-env's lane_distance_to is a center-to-center distance and these
# vehicles are about 5 m long. The old 5 m standstill gap left no bumper
# clearance. This baseline uses 8 m; V1 keeps at least 1 m geometric margin.
NL3_V0 = SUTProfile("nl3_v0", "IDM", time_wanted=1.5, desired_gap=8.0,
                    comfort_acceleration=2.5, max_brake=5.0)
NL3_V1 = replace(NL3_V0, name="nl3_v1", time_wanted=1.0,
                 desired_gap=6.0, comfort_acceleration=2.8)
NL3_V2 = replace(NL3_V1, name="nl3_v2", emergency_ttc=3.0,
                 emergency_brake_gain=8.0, emergency_max_brake=11.0)
NL3_RELEASES = {profile.name: profile for profile in (NL3_V0, NL3_V1, NL3_V2)}
NL3_PARENTS = {"nl3_v0": None, "nl3_v1": "nl3_v0", "nl3_v2": "nl3_v1"}
NL3_PURPOSES = {
    "nl3_v0": "Geometrically separated IDM reference baseline",
    "nl3_v1": "Efficiency calibration preserving standstill bumper clearance",
    "nl3_v2": "Short-TTC predictive-brake safeguard inherited from nl3_v1",
}
