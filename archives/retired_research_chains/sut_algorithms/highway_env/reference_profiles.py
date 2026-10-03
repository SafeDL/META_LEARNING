"""Controller profiles still used by simulator adapters and smoke tests."""

from sut_algorithms.highway_env.idm_profiles import ADATE_SOURCE_PROFILES, SUTProfile


IDM_REFERENCE_PROFILE = SUTProfile(
    "idm_ref", "IDM", time_wanted=1.5, max_brake=5.0,
    reaction_delay=0.0, desired_gap=5.0,
    comfort_acceleration=3.0, target_speed=27.0,
)
FVDM_REFERENCE_PROFILE = next(
    profile for profile in ADATE_SOURCE_PROFILES
    if profile.name == "SM-Strong-FVDM"
)
