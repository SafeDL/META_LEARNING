"""Shared, fixed experiment settings; entry points need no CLI arguments."""
from dataclasses import asdict
from pathlib import Path

from highway_sim_env.build_spec import BuildSpec
from sut_algorithms.highway_env.perception import PerceptionProfile


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/history_guided_testing"
SEEDS = (11, 23, 37, 53, 71)
BUDGET = 200
CHECKPOINTS = (10, 30, 50, 100, 150, 200)
WORKERS = 6
TARGET_SAMPLING_SEEDS = (75019, 75020)
GROUPS = {"reactive": ("idm_reactive", "fvdm_reactive"),
          "delay": ("idm_delayed", "fvdm_delayed"),
          "limited": ("idm_limited",), "predictive": ("idm_predictive",)}
PROFILES = (
    PerceptionProfile("idm_reactive", "IDM", max_brake=6, desired_gap=6, target_speed=23),
    PerceptionProfile("fvdm_reactive", "FVDM", max_brake=6, desired_gap=6, target_speed=23,
                      fvdm_velocity_gain=1, fvdm_transition_gap=8),
    PerceptionProfile("idm_delayed", "IDM", max_brake=6, desired_gap=6, target_speed=23,
                      perception_delay_s=.6),
    PerceptionProfile("fvdm_delayed", "FVDM", max_brake=6, desired_gap=6, target_speed=23,
                      fvdm_velocity_gain=1, fvdm_transition_gap=8, perception_delay_s=.6),
    PerceptionProfile("idm_limited", "IDM", max_brake=3, desired_gap=6, target_speed=23),
    PerceptionProfile("idm_predictive", "IDM", max_brake=6, desired_gap=6, target_speed=23,
                      perception_mode="constant_velocity", prediction_horizon_s=2,
                      emergency_ttc=2.5, emergency_brake_gain=3, emergency_max_brake=6),
)
TARGET = PerceptionProfile(
    "fvdm_target", "FVDM", max_brake=8, desired_gap=8, target_speed=23,
    fvdm_sensitivity=.5, fvdm_velocity_gain=.8, fvdm_transition_gap=8,
    perception_delay_s=.15,
)
TEMPLATES = ("fbrt_cutin", "fbrt_lead_emergency_brake")
BOUNDS = (((8, 60), (15, 25), (1.5, 3), (.5, 2)),
          ((8, 70), (15, 25), (.5, 7), (.5, 2)))
MEASUREMENT = {"ttc_scale_s": 1.5, "drac_scale_mps2": 3., "gap_scale_m": 1.,
               "ttc_prediction_horizon_s": 6.}
FEEDBACK = {"noise_variance": .0025, "radius": .5}
TRAINING = {"steps": 1600, "validation_interval": 200, "lr": .0003,
            "query_count": 48, "context_count": 256, "support_counts": (0, 5, 10, 20, 50, 100)}
BASELINES = {"mean": .5, "variance": .25, "lengthscale": .2,
             "noise_variance": .001, "jitter": 1e-8, "initial_queries": 10,
             "ucb_delta": .1, "bas_threshold": .5, "rf_trees": 50,
             "rf_max_depth": 10, "rf_min_samples_split": 2, "rf_jackknife_folds": 10}


def build_spec(profile):
    return BuildSpec(profile.name, "profiled_" + profile.controller.lower(), None,
                     "legacy_profile", profile.controller, 20., profile=asdict(profile))


def group_of(name):
    return next(group for group, names in GROUPS.items() if name in names)
