"""Machine-readable G1--G8 diagnostics used before MVR experiments."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def gate_map_representation(encoded_families: Sequence[str], cache_max_abs_error: float, se2_max_abs_error: float) -> dict[str, object]:
    return {"gate": "G1", "pass": set(encoded_families) >= {"merge", "cutin", "roundabout"} and cache_max_abs_error <= 1e-6 and se2_max_abs_error <= 1e-5, "families": list(encoded_families), "cache_max_abs_error": cache_max_abs_error, "se2_max_abs_error": se2_max_abs_error}


def gate_scenario_execution(valid_resets: int, attempted_resets: int, invalid_fail_fast: bool) -> dict[str, object]:
    return {"gate": "G2", "pass": attempted_resets > 0 and valid_resets == attempted_resets and invalid_fail_fast, "valid_resets": valid_resets, "attempted_resets": attempted_resets, "invalid_fail_fast": invalid_fail_fast}


def gate_heterogeneity(within_sut_score: float, cross_sut_score: float) -> dict[str, object]:
    return {"gate": "G3", "pass": cross_sut_score > within_sut_score, "within_sut_score": within_sut_score, "cross_sut_score": cross_sut_score}


def gate_sut_heterogeneity(mean_failure_disagreement: float, mean_severity_distance: float, valid_rate: float, failure_rate: float) -> dict[str, object]:
    """Assess aligned scenario outcomes across meta-train IDM profiles."""
    return {
        "gate": "G3",
        "pass": valid_rate >= 0.80 and 0.02 <= failure_rate <= 0.80 and mean_failure_disagreement >= 0.10,
        "mean_failure_disagreement": mean_failure_disagreement,
        "mean_severity_distance": mean_severity_distance,
        "valid_rate": valid_rate,
        "failure_rate": failure_rate,
    }


def gate_inner_controllability(option_effect_size: float, repeated_success_rate: float) -> dict[str, object]:
    return {"gate": "G4", "pass": option_effect_size > 0.0 and repeated_success_rate >= 0.8, "option_effect_size": option_effect_size, "repeated_success_rate": repeated_success_rate}


def gate_identifiability(within_distance: float, between_distance: float) -> dict[str, object]:
    return {"gate": "G5", "pass": between_distance > within_distance, "within_sut_distance": within_distance, "between_sut_distance": between_distance}


def gate_posterior_utility(correct: float, swapped: float, zero: float) -> dict[str, object]:
    return {"gate": "G6", "pass": correct > max(swapped, zero), "correct_z": correct, "swapped_z": swapped, "zero_z": zero}


def gate_outer_utility(outer_with_z: float, alternatives: Mapping[str, float]) -> dict[str, object]:
    return {"gate": "G7", "pass": bool(alternatives) and outer_with_z > max(alternatives.values()), "outer_with_z": outer_with_z, "alternatives": dict(alternatives)}


def gate_all_in_budget(full_curve: Sequence[float], baseline_curve: Sequence[float]) -> dict[str, object]:
    if len(full_curve) != len(baseline_curve) or not full_curve:
        raise ValueError("fixed-budget curves must be non-empty and aligned")
    full_auc, baseline_auc = float(np.trapz(full_curve)), float(np.trapz(baseline_curve))
    return {"gate": "G8", "pass": full_auc > baseline_auc, "full_auc": full_auc, "baseline_auc": baseline_auc, "all_in_budget": True}
