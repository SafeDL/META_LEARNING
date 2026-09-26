"""Static capability resolution and deterministic scenario catalogue compiler."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.vehicle.controller import MDPVehicle
from highway_env_benchmark.envs.external_cutin import ExternalCutInEnv


ROOT = Path("results/method_chains/failure_memory_regression/memory_v2")
CATALOGUE = Path("configs/fbrt/scenario_catalogue_v2.yaml")
PPO_CHECKPOINT = Path("assets/ppo_ece/vd_1_5_trial_1.zip")
RECIPES = {
    "legacy_core_5": ["S01", "S02", "S03", "S04", "S05"],
    "highway_policy_compatible_5": ["S01", "S02", "S05", "S06", "S08"],
}


def load_catalogue(path: Path = CATALOGUE) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"scenario catalogue is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "fbrt_scenario_catalogue_v2":
        raise ValueError("unsupported scenario catalogue schema")
    if len(payload.get("scenarios", [])) != 14:
        raise ValueError("v2 catalogue must retain all 14 candidate cards")
    return payload


def static_capabilities() -> dict:
    speeds = [float(x) for x in np.asarray(MDPVehicle.DEFAULT_TARGET_SPEEDS).tolist()]
    action_names = [DiscreteMetaAction.ACTIONS_ALL[index]
                    for index in sorted(DiscreteMetaAction.ACTIONS_ALL)]
    policy_config = ExternalCutInEnv.default_config()
    observation_config = policy_config["observation"]
    checkpoint = PPO_CHECKPOINT if PPO_CHECKPOINT.is_absolute() else Path.cwd() / PPO_CHECKPOINT
    return {
        "installed_highway_env_version": getattr(__import__("highway_env"), "__version__", "unknown"),
        "policy_observation_shape": [int(observation_config["vehicles_count"]),
                                     len(observation_config["features"])],
        "policy_observation_contract": "ExternalCutInEnv Kinematics; normalized; relative; sorted; see_behind",
        "policy_observation_config": observation_config,
        "policy_action_names": action_names,
        "policy_target_speeds_mps": speeds,
        "minimum_commandable_speed_mps": min(speeds),
        "can_command_full_stop": min(speeds) <= 0.0,
        "can_change_lane": "LANE_LEFT" in action_names or "LANE_RIGHT" in action_names,
        "supported_road_topology": ["straight_1_lane", "straight_2_lane"],
        "ppo_checkpoint_path": str(PPO_CHECKPOINT.as_posix()),
        "ppo_checkpoint_available": checkpoint.is_file(),
        "ppo_runtime_dependency_available": importlib.util.find_spec("stable_baselines3") is not None,
        "ppo_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint.is_file() else None,
        "native_mobil_can_command_full_stop": True,
        "native_mobil_can_change_lane": True,
        "legacy_profiled_idm_can_command_full_stop": True,
        "legacy_profiled_idm_can_change_lane": False,
    }


def _existing_recipe(root: Path = ROOT) -> str | None:
    protocol_path = root / "protocol.json"
    episode_path = root / "compact_bank" / "episodes.jsonl"
    if not protocol_path.is_file() or not episode_path.is_file():
        return None
    if not episode_path.read_text(encoding="utf-8").strip():
        return None
    try:
        return json.loads(protocol_path.read_text(encoding="utf-8")).get("selected_recipe")
    except json.JSONDecodeError:
        return None


def resolve_recipe(capability: dict, root: Path = ROOT) -> tuple[str, str]:
    existing = _existing_recipe(root)
    if existing in RECIPES:
        return existing, "existing_compact_bank_frozen"
    if not capability["can_command_full_stop"]:
        return "highway_policy_compatible_5", "static_action_capability_no_full_stop"
    return "legacy_core_5", "static_action_capability_supports_full_stop"


def write_capability_and_recipe(root: Path = ROOT) -> tuple[dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    catalogue = load_catalogue()
    capability = static_capabilities()
    recipe, reason = resolve_recipe(capability, root)
    scenario_by_id = {item["id"]: item for item in catalogue["scenarios"]}
    selected = [scenario_by_id[key] for key in RECIPES[recipe]]
    builds = catalogue["new_builds"]
    rows = []
    for build in builds:
        kind = "native_mobil" if build.startswith("mobil_") else "ppo"
        for item in catalogue["scenarios"]:
            required = set(item.get("required_capability", []))
            can_stop = capability["native_mobil_can_command_full_stop"] if kind == "native_mobil" else capability["can_command_full_stop"]
            can_lane = capability["native_mobil_can_change_lane"] if kind == "native_mobil" else capability["can_change_lane"]
            applicable = True
            reason_text = ""
            if "full_stop" in required and not can_stop:
                applicable, reason_text = False, "NOT_APPLICABLE_ACTION_CAPABILITY"
            if "full_stop_or_lateral_escape" in required and not (can_stop or can_lane):
                applicable, reason_text = False, "NOT_APPLICABLE_ACTION_CAPABILITY"
            if "autonomous_lane_change" in required and not can_lane:
                applicable, reason_text = False, "NOT_APPLICABLE_ACTION_CAPABILITY"
            if item["id"] in RECIPES[recipe] and kind == "ppo" and not capability["ppo_checkpoint_available"]:
                reason_text = "PPO_UNAVAILABLE_CHECKPOINT_MISSING" if not reason_text else reason_text + ";PPO_UNAVAILABLE_CHECKPOINT_MISSING"
            rows.append({"build_id": build, "adapter_kind": kind,
                         "scenario_id": item["id"], "template_id": item["template_id"],
                         "selected": item["id"] in RECIPES[recipe],
                         "applicable_by_static_capability": applicable,
                         "execution_status": "selected" if item["id"] in RECIPES[recipe] else "not_selected",
                         "reason": reason_text})
    with (root / "capability_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_payload = {
        "recipe": recipe, "selection_reason": reason,
        "selection_frozen_before_method_comparison": True,
        "selected_scenario_ids": RECIPES[recipe],
        "max_scenarios": 5, "samples_per_template": int(catalogue["samples_per_template"]),
        "main_bank_episode_cap": 320,
        "ppo_checkpoint_status": "available" if capability["ppo_checkpoint_available"] else "PPO_UNAVAILABLE",
        "target_outcome_filtering": False,
        "all_builds_share_candidates": True,
    }
    (root / "capabilities.json").write_text(
        json.dumps(capability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "selected_recipe.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return capability, selected_payload


def _sample_parameters(item: dict, seed: int) -> list[dict]:
    names = list(item["research_bounds"])
    bounds = np.asarray([item["research_bounds"][name] for name in names], dtype=float)
    template_index = int(item["id"][1:])
    sobol = qmc.Sobol(d=2, scramble=True, seed=seed + template_index).random_base2(m=4)
    samples = [("anchor_like", (0.25, 0.25)), ("anchor_like", (0.25, 0.75)),
               ("anchor_like", (0.75, 0.25)), ("anchor_like", (0.75, 0.75))]
    samples += [("space_filling", (float(x), float(y))) for x, y in sobol[:12]]
    output = []
    for index, (kind, unit) in enumerate(samples):
        values = {name: float(bounds[i, 0] + unit[i] * (bounds[i, 1] - bounds[i, 0]))
                  for i, name in enumerate(names)}
        scenario_id = f"{seed}:{item['id']}:{kind}:{index}"
        compiled = {"scenario_id": scenario_id, "template_id": item["template_id"],
                    "catalogue_id": item["id"], "parameterization_version": item["parameterization_version"],
                    "sampling_kind": kind, "active_parameters": values,
                    "fixed_context": dict(item["fixed_context"]),
                    "research_bounds": dict(item["research_bounds"]),
                    "evidence": list(item.get("evidence", [])),
                    "range_status": "RESEARCH_RANGE",
                    "required_capability": list(item.get("required_capability", [])),
                    "context_id": f"{item['id']}:research_v2:" + hashlib.sha256(
                        json.dumps(item["fixed_context"], sort_keys=True).encode()).hexdigest()[:10]}
        output.append(compiled)
    return output


def _validate_geometry(case: dict) -> list[str]:
    errors = []
    active = case["active_parameters"]
    context = case["fixed_context"]
    template = case["template_id"]
    if template == "fbrt_lane_change_rear":
        if active["rear_clearance_m"] <= 0:
            errors.append("rear_vehicle_initial_overlap")
        if context["ego_speed_mps"] + active["rear_closing_speed_mps"] > 35:
            errors.append("rear_speed_exceeds_road_odD")
    if template in {"fbrt_cutin", "fbrt_cutout_static", "fbrt_cutin_then_brake"}:
        if active["initial_clearance_m"] <= 0:
            errors.append("lead_initial_overlap")
    return errors


def compile_scenarios(selected_recipe: dict, root: Path = ROOT,
                      seed: int = 4179901) -> list[dict]:
    catalogue = load_catalogue()
    chosen = set(selected_recipe["selected_scenario_ids"])
    items = [row for row in catalogue["scenarios"] if row["id"] in chosen]
    cases = [case for item in items for case in _sample_parameters(item, seed)]
    errors = [(case["scenario_id"], error) for case in cases for error in _validate_geometry(case)]
    if errors:
        raise ValueError(f"pre-simulation geometry validation failed: {errors}")
    root.mkdir(parents=True, exist_ok=True)
    case_path = root / "compact_bank" / "scenario_cases.jsonl"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                     for row in cases), encoding="utf-8")
    manifest_rows = []
    for case in cases:
        for name, value in case["active_parameters"].items():
            bounds = case["research_bounds"][name]
            manifest_rows.append({"scenario_id": case["scenario_id"],
                                  "catalogue_id": case["catalogue_id"],
                                  "template_id": case["template_id"],
                                  "parameter_name": name, "value": value, "unit":
                                  "m/s^2" if "deceleration" in name else
                                  "m/s" if "speed" in name else
                                  "s" if name.endswith("_s") or "ttc" in name else "m",
                                  "lower_bound": bounds[0], "upper_bound": bounds[1],
                                  "range_status": case["range_status"],
                                  "sampling_kind": case["sampling_kind"],
                                  "fixed_context": json.dumps(case["fixed_context"], sort_keys=True),
                                  "context_id": case["context_id"],
                                  "source_ids": ";".join(case["evidence"])})
    with (root / "scene_parameter_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return cases
