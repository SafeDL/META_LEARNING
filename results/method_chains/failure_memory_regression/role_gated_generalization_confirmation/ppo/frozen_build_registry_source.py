"""Registry for retained Highway-env systems under test."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import Policy
from .idm_mobil import IDMMobilPolicy
from .mcts_cv import MCTSCVPolicy
from .ppo_ece import PPO_CHECKPOINT, PPOPolicy
from .value_iteration import ValueIterationPolicy

RETAINED_SUTS = ("idm_mobil", "vi_ttc", "mcts_cv", "ppo_ece")
PPO_RELEASE_CHECKPOINTS = Path(
    "results/method_chains/failure_memory_regression/ppo_release/checkpoints")


def policy_factory(sut: str, assets_root: Path) -> Policy:
    if sut == "idm_mobil":
        return IDMMobilPolicy()
    if sut == "vi_ttc":
        return ValueIterationPolicy()
    if sut == "mcts_cv":
        return MCTSCVPolicy()
    if sut == "ppo_ece":
        checkpoint = assets_root / "ppo_ece" / "vd_1_5_trial_1.zip"
        return PPOPolicy(checkpoint)
    raise KeyError(f"Unsupported SUT: {sut}")


def build_spec_factory(build_id: str):
    """Resolve the unified FBRT build description without changing legacy factories."""
    from methods.failure_memory_regression.schema import BuildSpec
    from sut_algorithms.highway_env.idm_profiles import SUTProfile

    if build_id in {"nl_v0", "nl_v1", "nl_v2"}:
        from sut_algorithms.highway_env.nl_release import NL_PARENTS, NL_RELEASES
        return BuildSpec(build_id, "profiled_idm_release", NL_PARENTS[build_id],
                         "legacy_profile", "Profiled-IDM", 20.0,
                         profile=NL_RELEASES[build_id].__dict__.copy())
    if build_id in {"nl_eff_c1", "nl_eff_c2", "nl_eff_c3"}:
        from sut_algorithms.highway_env.nl_release import NL_EFFICIENCY_CANDIDATES
        return BuildSpec(build_id, "profiled_idm_calibration", "nl_v0",
                         "legacy_profile", "Profiled-IDM", 20.0,
                         profile=NL_EFFICIENCY_CANDIDATES[build_id].__dict__.copy())
    if build_id in {"nl2_v1", "nl2_v2"}:
        from sut_algorithms.highway_env.nl_release import NL2_PARENTS, NL2_RELEASES
        return BuildSpec(build_id, "profiled_idm_release", NL2_PARENTS[build_id],
                         "legacy_profile", "Profiled-IDM", 20.0,
                         profile=NL2_RELEASES[build_id].__dict__.copy())
    if build_id in {"nl3_v0", "nl3_v1", "nl3_v2"}:
        from sut_algorithms.highway_env.nl_release import NL3_PARENTS, NL3_RELEASES
        return BuildSpec(build_id, "profiled_idm_release", NL3_PARENTS[build_id],
                         "legacy_profile", "Profiled-IDM", 20.0,
                         profile=NL3_RELEASES[build_id].__dict__.copy())

    if build_id == "idm_ref":
        from methods.core_mine.idm_revision_pilot import REFERENCE
        reference = REFERENCE if REFERENCE.name == build_id else SUTProfile("idm_ref", "IDM")
        profile = reference.__dict__.copy()
        return BuildSpec(build_id, "profiled_idm", None, "legacy_profile",
                         "Profiled-IDM", 20.0, profile=profile)
    if build_id in {"merge_blind06", "merge_brake2", "slow_front_brake2"}:
        return BuildSpec(build_id, "profiled_idm_faults", "idm_ref", "legacy_profile",
                         "Profiled-IDM", 20.0, profile=SUTProfile("legacy", "IDM").__dict__.copy(),
                         mutation={"legacy_fault": build_id})
    if build_id in {"mobil_ref_v2", "mobil_rear_guard_off_v2",
                    "mobil_rear_state_age", "mobil_rear_state_age080"}:
        if build_id == "mobil_ref_v2":
            mutation = None
        elif build_id == "mobil_rear_guard_off_v2":
            mutation = {
                "rear_guard": "off",
                "guard_condition": "new_following_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED",
            }
        else:
            mutation = {
                "rear_state_age_s": {"mobil_rear_state_age": 0.30,
                                     "mobil_rear_state_age080": 0.80}[build_id],
                "scope": "candidate_lane_rear_predicted_braking_only",
            }
        return BuildSpec(build_id, "native_idm_mobil", None if not mutation else "mobil_ref_v2",
                         "native_vehicle", "highway-env IDM+MOBIL", 20.0,
                         mutation=mutation)
    if build_id in {"ppo_ref_v2", "ppo_obs_age020_v2"}:
        checkpoint = PPO_CHECKPOINT
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
        mutation = None if build_id == "ppo_ref_v2" else {"observation_delay_s": 0.20}
        return BuildSpec(build_id, "ppo_ece", None if not mutation else "ppo_ref_v2",
                         "external_meta_policy", "PPO-ECE", 5.0,
                         profile={"checkpoint_path": str(checkpoint)},
                         checkpoint_sha256=digest, mutation=mutation)
    if build_id in {"ppo_release_v0", "ppo_release_v1", "ppo_release_v2"}:
        parent = {"ppo_release_v0": None, "ppo_release_v1": "ppo_release_v0",
                  "ppo_release_v2": "ppo_release_v1"}[build_id]
        checkpoint = PPO_RELEASE_CHECKPOINTS / f"{build_id}.zip"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
        return BuildSpec(build_id, "ppo_weight_release", parent,
                         "external_meta_policy", "PPO-ECE-continued", 5.0,
                         profile={"checkpoint_path": str(checkpoint)},
                         checkpoint_sha256=digest)
    if build_id == "vi_ttc_ref_audit_v4":
        return BuildSpec(build_id, "vi_ttc", None, "external_meta_policy",
                         "VI-TTC", 5.0)
    if build_id == "mcts_cv_ref_audit_v4":
        return BuildSpec(build_id, "mcts_cv", None, "external_meta_policy",
                         "MCTS-CV", 5.0)
    raise KeyError(f"Unsupported FBRT build: {build_id}")
