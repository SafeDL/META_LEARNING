"""Convert runner transitions into the shared outcome and signature schema."""
from __future__ import annotations

from typing import Any, Mapping

from .criteria import FailureCriteria
from .signature import FailureSignature, FailureSignatureBuilder


def analyze_rollout(
    transitions: list[dict[str, Any]],
    scenario_family: str,
    conflict_zone_id: str,
    candidate_id: str,
    criteria: FailureCriteria,
) -> tuple[Mapping[str, Any], FailureSignature]:
    if not transitions:
        raise ValueError("cannot analyze an empty rollout")
    infos = [row["info"] for row in transitions]
    features = [row["trajectory_features"] for row in transitions]
    outcome = {
        "non_target_collision": any(bool(info.get("non_target_collision", False)) for info in infos),
        "adversary_out_of_road": any(bool(info.get("adversary_out_of_road", False)) for info in infos),
        "sut_out_of_road": any(bool(info.get("sut_out_of_road", False)) for info in infos),
        "wrong_route": any(bool(info.get("wrong_route", False)) for info in infos),
        "adversary_traffic_violation": any(
            bool(info.get("adversary_traffic_violation", False)) for info in infos
        ),
        "target_collision": any(bool(info.get("target_collision", info.get("crash_vehicle", False))) for info in infos),
        "min_ttc": min(float(row[8]) * 15.0 for row in features),
        "min_distance": min(float(row[10]) * 100.0 for row in features),
        "max_closing_speed": max(float(row[10]) * 100.0 / max(float(row[8]) * 15.0, 1e-3) if row[8] < 1.0 else 0.0 for row in features),
    }
    final_info = infos[-1]
    outcome["test_completion_condition"] = final_info.get("test_completion_condition")
    outcome["termination_reason"] = final_info.get("termination_reason")
    outcome["sut_arrived_destination"] = any(
        bool(info.get("sut_arrived_destination", False)) for info in infos
    )
    outcome["test_process_completed"] = bool(outcome["sut_arrived_destination"])
    event_infos = [info for info in infos if info.get("event_kind") is not None]
    event_info = event_infos[0] if event_infos else {}
    outcome["event_kind"] = event_info.get("event_kind")
    outcome["event_semantic_valid"] = bool(event_info.get("event_semantic_valid", False))
    outcome["event_traffic_valid"] = bool(event_info.get("event_traffic_valid", False))
    traffic_infos = [info for info in infos if "traffic_base_action" in info]
    if traffic_infos:
        final_traffic = traffic_infos[-1]
        outcome["traffic_telemetry"] = {
            "base_actions": [list(info["traffic_base_action"]) for info in traffic_infos],
            "candidate_actions": [list(info["traffic_candidate_action"]) for info in traffic_infos],
            "executed_actions": [list(info["traffic_executed_action"]) for info in traffic_infos],
            "shield_intervention_l2": [
                float(info["traffic_shield_intervention_l2"]) for info in traffic_infos
            ],
            "rejection_counts": dict(final_traffic["traffic_rejection_counts"]),
            "violation_counts": dict(final_traffic["traffic_violation_counts"]),
            "max_speed_mps": float(final_traffic["traffic_max_speed_mps"]),
            "max_abs_acceleration_mps2": float(final_traffic["traffic_max_abs_acceleration_mps2"]),
            "max_abs_jerk_mps3": float(final_traffic["traffic_max_abs_jerk_mps3"]),
            "max_lateral_acceleration_mps2": float(final_traffic["traffic_max_lateral_acceleration_mps2"]),
            "legal_lane_lateral_m": float(final_traffic["traffic_legal_lane_lateral_m"]),
        }
    sut_infos = [info for info in infos if "sut_steering" in info]
    if sut_infos:
        outcome["sut_telemetry"] = {
            "current_lanes": [list(info["sut_current_lane"]) for info in sut_infos],
            "routing_target_lanes": [
                None if info["sut_routing_target_lane"] is None else list(info["sut_routing_target_lane"])
                for info in sut_infos
            ],
            "current_ref_lanes": [
                [list(lane) for lane in info["sut_current_ref_lanes"]] for info in sut_infos
            ],
            "steering": [float(info["sut_steering"]) for info in sut_infos],
            "lateral_error_m": [float(info["sut_lateral_error_m"]) for info in sut_infos],
            "heading_error_rad": [float(info["sut_heading_error_rad"]) for info in sut_infos],
            "speed_mps": [float(info["sut_speed_mps"]) for info in sut_infos],
            "route_progress_m": [float(info["sut_route_progress_m"]) for info in sut_infos],
            "target_speed_mps": [float(info["sut_target_speed_mps"]) for info in sut_infos],
            "nominal_target_speed_mps": [
                float(info["sut_nominal_target_speed_mps"]) for info in sut_infos
            ],
            "curve_safe_speed_mps": [float(info["sut_curve_safe_speed_mps"]) for info in sut_infos],
        }
    invalid = any(
        bool(outcome[key])
        for key in (
            "non_target_collision",
            "adversary_out_of_road",
            "sut_out_of_road",
            "wrong_route",
            "adversary_traffic_violation",
        )
    )
    event_valid = bool(
        outcome["event_semantic_valid"] and outcome["event_traffic_valid"]
    )
    outcome["valid_target_collision"] = bool(
        outcome["target_collision"]
        and outcome["event_kind"] == "collision"
        and event_valid
    )
    outcome["valid_critical_near_miss"] = (
        outcome["event_kind"] == "near_miss" and event_valid
    )
    # A valid event is frozen before post-impact simulator transients.  For
    # episodes without an event, ordinary traffic validity remains decisive.
    invalid = invalid and not (
        outcome["valid_target_collision"] or outcome["valid_critical_near_miss"]
    )
    outcome["is_valid_episode"] = not invalid
    signature = FailureSignatureBuilder(criteria).from_outcome(
        outcome, scenario_family, conflict_zone_id, candidate_id
    )
    outcome.update({"is_valid_episode": signature.is_valid_episode, "is_failure": signature.is_failure, "is_collision": bool(outcome["valid_target_collision"]), "is_near_miss": bool(outcome["valid_critical_near_miss"]), "severity_vector": signature.severity_vector, "candidate_id": candidate_id, "conflict_zone_id": conflict_zone_id, "failure_signature": signature.signature_id})
    return outcome, signature
