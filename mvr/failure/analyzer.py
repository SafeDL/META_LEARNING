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
    traffic_infos = [info for info in infos if "traffic_raw_action" in info]
    if traffic_infos:
        final_traffic = traffic_infos[-1]
        outcome["traffic_telemetry"] = {
            "raw_actions": [list(info["traffic_raw_action"]) for info in traffic_infos],
            "applied_actions": [list(info["traffic_applied_action"]) for info in traffic_infos],
            "rejection_counts": dict(final_traffic["traffic_rejection_counts"]),
            "violation_counts": dict(final_traffic["traffic_violation_counts"]),
            "max_speed_mps": float(final_traffic["traffic_max_speed_mps"]),
            "max_abs_acceleration_mps2": float(final_traffic["traffic_max_abs_acceleration_mps2"]),
            "max_abs_jerk_mps3": float(final_traffic["traffic_max_abs_jerk_mps3"]),
            "max_lateral_acceleration_mps2": float(final_traffic["traffic_max_lateral_acceleration_mps2"]),
            "max_abs_lane_lateral_m": float(final_traffic["traffic_max_abs_lane_lateral_m"]),
            "lane_change_completed": bool(final_traffic["traffic_lane_change_completed"]),
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
    outcome["valid_target_collision"] = bool(outcome["target_collision"] and not invalid)
    outcome["valid_critical_near_miss"] = (
        not outcome["target_collision"]
        and not invalid
        and outcome["min_ttc"] < criteria.ttc_s
        and outcome["min_distance"] < criteria.distance_m
    )
    signature = FailureSignatureBuilder(criteria).from_outcome(
        outcome, scenario_family, conflict_zone_id, candidate_id
    )
    outcome.update({"is_valid_episode": signature.is_valid_episode, "is_failure": signature.is_failure, "is_collision": bool(outcome["target_collision"]), "is_near_miss": bool(outcome["valid_critical_near_miss"]), "severity_vector": signature.severity_vector, "candidate_id": candidate_id, "conflict_zone_id": conflict_zone_id, "failure_signature": signature.signature_id})
    return outcome, signature
