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

from highway_sim_env.envs.external_cutin import ExternalCutInEnv
from sut_algorithms.highway_env.ppo_ece import PPO_CHECKPOINT


ROOT = Path("results/method_chains/failure_memory_regression/memory_v2")
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CATALOGUE = CONFIG_DIR / "scenario_catalogue.yaml"
INTERACTION_CATALOGUE = CONFIG_DIR / "interaction_catalogue.yaml"
INTERACTION_HOLDOUT_CATALOGUE = CONFIG_DIR / "interaction_holdout_catalogue.yaml"
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
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return protocol.get("selected_recipe")


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
    selected_ids = set(RECIPES[recipe])
    builds = catalogue["new_builds"]
    rows = []
    for build in builds:
        kind = "native_mobil" if build.startswith("mobil_") else "ppo"
        if kind == "native_mobil":
            can_stop = capability["native_mobil_can_command_full_stop"]
            can_lane = capability["native_mobil_can_change_lane"]
        else:
            can_stop = capability["can_command_full_stop"]
            can_lane = capability["can_change_lane"]
        for item in catalogue["scenarios"]:
            required = set(item.get("required_capability", []))
            selected = item["id"] in selected_ids
            applicable = True
            reason_text = ""
            if "full_stop" in required and not can_stop:
                applicable, reason_text = False, "NOT_APPLICABLE_ACTION_CAPABILITY"
            if "full_stop_or_lateral_escape" in required and not (can_stop or can_lane):
                applicable, reason_text = False, "NOT_APPLICABLE_ACTION_CAPABILITY"
            if "autonomous_lane_change" in required and not can_lane:
                applicable, reason_text = False, "NOT_APPLICABLE_ACTION_CAPABILITY"
            if selected and kind == "ppo" and not capability["ppo_checkpoint_available"]:
                reason_text = "PPO_UNAVAILABLE_CHECKPOINT_MISSING" if not reason_text else reason_text + ";PPO_UNAVAILABLE_CHECKPOINT_MISSING"
            rows.append({"build_id": build, "adapter_kind": kind,
                         "scenario_id": item["id"], "template_id": item["template_id"],
                         "selected": selected,
                         "applicable_by_static_capability": applicable,
                         "execution_status": "selected" if selected else "not_selected",
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


def _sample_parameters(item: dict, seed: int, sample_count: int = 16) -> list[dict]:
    names = list(item["research_bounds"])
    bounds = np.asarray([item["research_bounds"][name] for name in names], dtype=float)
    dimension = len(names)
    if dimension < 1 or bounds.shape != (dimension, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(f"invalid active parameter bounds for {item['id']}")
    # Retain the exact historical two-dimensional stream for the frozen v2 bank.
    template_index = (int(item["id"][1:]) if item["id"].startswith("S")
                      else int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16))
    if sample_count < 4 or sample_count & (sample_count - 1):
        raise ValueError("sample count must be a power of two >= 4")
    sobol = qmc.Sobol(d=dimension, scramble=True,
                      seed=seed + template_index).random_base2(m=int(np.log2(sample_count)))
    if dimension == 2:
        anchors = [(0.25, 0.25), (0.25, 0.75),
                   (0.75, 0.25), (0.75, 0.75)]
    else:
        anchors = [tuple(0.25 if (index >> (axis % 2)) & 1 == 0 else 0.75
                         for axis in range(dimension)) for index in range(4)]
    samples = [("anchor_like", unit) for unit in anchors]
    samples += [("space_filling", tuple(float(x) for x in unit))
                for unit in sobol[:sample_count - 4]]
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
        if "context_id" in item:
            compiled["context_id"] = str(item["context_id"])
        if "interaction_family" in item:
            compiled["interaction_family"] = item["interaction_family"]
            compiled["actor_roles"] = list(item["actor_roles"])
        output.append(compiled)
    return output


def compile_interaction_scenarios(root: Path, seed: int = 4179901,
                                  catalogue_path: Path = INTERACTION_CATALOGUE) -> list[dict]:
    """Compile the independent A/B interaction contract without touching v2 banks."""
    payload = yaml.safe_load(catalogue_path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {"fbrt_interaction_catalogue_v1",
                                     "fbrt_interaction_catalogue_v2"}:
        raise ValueError("unsupported interaction catalogue")
    items = payload.get("scenarios", [])
    if [item["id"] for item in items] != ["IA", "IB"]:
        raise ValueError("interaction catalogue must contain exactly IA and IB")
    count = int(payload.get("samples_per_preset", 16))
    cases = [case for item in items for case in _sample_parameters(item, seed, count)]
    errors = [(case["scenario_id"], error) for case in cases for error in _validate_geometry(case)]
    if errors:
        raise ValueError(f"interaction geometry validation failed: {errors}")
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": payload["schema"],
        "catalogue_sha256": hashlib.sha256(catalogue_path.read_bytes()).hexdigest(),
        "seed": seed,
        "candidate_count": len(cases),
        "candidate_sha256": hashlib.sha256(json.dumps(
            cases, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "target_outcome_filtering": False,
    }
    protocol_path = root / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError(f"interaction protocol changed after freezing: {root}")
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    (root / "scenario_cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8")
    return cases


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
    if template in {"fbrt_interaction_front_rear", "fbrt_interaction_cutin_escape"}:
        front = active.get("front_clearance_m", active.get("initial_front_clearance_m", 0))
        if front <= 0 or active["rear_clearance_m"] <= 0:
            errors.append("interaction_initial_overlap")
        if float(context["ego_speed_mps"]) + active["rear_closing_speed_mps"] > 35:
            errors.append("rear_speed_exceeds_limit")
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
