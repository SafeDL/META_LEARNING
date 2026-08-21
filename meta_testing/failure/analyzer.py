"""Convert runner transitions into the shared outcome and signature schema."""
from __future__ import annotations

from typing import Any, Mapping

from .signature import FailureSignature, FailureSignatureBuilder


def analyze_rollout(transitions: list[dict[str, Any]], scenario_family: str, conflict_zone_id: str, candidate_id: str) -> tuple[Mapping[str, Any], FailureSignature]:
    if not transitions:
        raise ValueError("cannot analyze an empty rollout")
    infos = [row["info"] for row in transitions]
    features = [row["trajectory_features"] for row in transitions]
    outcome = {
        "non_target_collision": any(bool(info.get("non_target_collision", False)) for info in infos),
        "adversary_out_of_road": any(bool(info.get("adversary_out_of_road", False)) for info in infos),
        "sut_out_of_road": any(bool(info.get("sut_out_of_road", False)) for info in infos),
        "wrong_route": any(bool(info.get("wrong_route", False)) for info in infos),
        "target_collision": any(bool(info.get("target_collision", info.get("crash_vehicle", False))) for info in infos),
        "min_ttc": min(float(row[8]) * 15.0 for row in features),
        "min_distance": min(float(row[10]) * 100.0 for row in features),
        "max_closing_speed": max(float(row[10]) * 100.0 / max(float(row[8]) * 15.0, 1e-3) if row[8] < 1.0 else 0.0 for row in features),
    }
    outcome["valid_critical_near_miss"] = not outcome["target_collision"] and outcome["min_ttc"] < 5.0 and outcome["min_distance"] < 10.0
    signature = FailureSignatureBuilder().from_outcome(outcome, scenario_family, conflict_zone_id, candidate_id)
    outcome.update({"is_valid_episode": signature.is_valid_episode, "is_failure": signature.is_failure, "is_collision": bool(outcome["target_collision"]), "is_near_miss": bool(outcome["valid_critical_near_miss"]), "severity_vector": signature.severity_vector, "candidate_id": candidate_id, "conflict_zone_id": conflict_zone_id, "failure_signature": signature.signature_id})
    return outcome, signature
