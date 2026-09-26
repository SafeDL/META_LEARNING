"""Registry for retained Highway-env systems under test."""

from __future__ import annotations

from pathlib import Path
import hashlib

from .base import Policy
from .fbrt_adapters import adapter_for
from .idm_mobil import IDMMobilPolicy
from .mcts_cv import MCTSCVPolicy
from .ppo_ece import PPOPolicy
from .value_iteration import ValueIterationPolicy

RETAINED_SUTS = ("idm_mobil", "vi_ttc", "mcts_cv", "ppo_ece")


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


def build_spec_factory(build_id: str, assets_root: Path = Path("assets")):
    """Resolve the unified FBRT build description without changing legacy factories."""
    from method_chains.failure_memory_regression.schema_v2 import BuildSpec
    from sut_algorithms.highway_env.idm_profiles import SUTProfile

    profile = None
    checkpoint = None
    if build_id == "idm_ref":
        from method_chains.core_mine.idm_revision_pilot import REFERENCE
        reference = REFERENCE if REFERENCE.name == build_id else SUTProfile("idm_ref", "IDM")
        profile = reference.__dict__.copy()
        return BuildSpec(build_id, "profiled_idm", None, "legacy_profile",
                         "Profiled-IDM", 20.0, profile=profile)
    if build_id in {"merge_blind06", "merge_brake2", "slow_front_brake2"}:
        return BuildSpec(build_id, "profiled_idm_faults", "idm_ref", "legacy_profile",
                         "Profiled-IDM", 20.0, profile=SUTProfile("legacy", "IDM").__dict__.copy(),
                         mutation={"legacy_fault": build_id})
    if build_id in {"mobil_ref_v2", "mobil_rear_guard_off_v2"}:
        mutation = None if build_id == "mobil_ref_v2" else {
            "rear_guard": "off",
            "guard_condition": "new_following_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED",
        }
        return BuildSpec(build_id, "native_idm_mobil", None if not mutation else "mobil_ref_v2",
                         "native_vehicle", "highway-env IDM+MOBIL", 20.0,
                         mutation=mutation)
    if build_id in {"ppo_ref_v2", "ppo_obs_age020_v2"}:
        checkpoint = assets_root / "ppo_ece" / "vd_1_5_trial_1.zip"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
        mutation = None if build_id == "ppo_ref_v2" else {"observation_delay_s": 0.20}
        return BuildSpec(build_id, "ppo_ece", None if not mutation else "ppo_ref_v2",
                         "external_meta_policy", "PPO-ECE", 5.0,
                         profile={"checkpoint_path": str(checkpoint)},
                         checkpoint_sha256=digest, mutation=mutation)
    if build_id == "vi_ttc_ref_audit_v4":
        return BuildSpec(build_id, "vi_ttc", None, "external_meta_policy",
                         "VI-TTC", 5.0)
    if build_id == "mcts_cv_ref_audit_v4":
        return BuildSpec(build_id, "mcts_cv", None, "external_meta_policy",
                         "MCTS-CV", 5.0)
    raise KeyError(f"Unsupported FBRT build: {build_id}")


def fbrt_adapter_factory(build_id: str, assets_root: Path = Path("assets")):
    """BuildSpec -> adapter route used by the unified FBRT execution contract."""
    return adapter_for(build_spec_factory(build_id, assets_root))
