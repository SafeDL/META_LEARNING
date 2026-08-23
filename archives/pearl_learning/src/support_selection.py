"""Leakage-safe ordering for a frozen support-case pool.

The selector intentionally consumes only fields available before an episode is
executed.  It is therefore suitable for comparing fixed, random and
initial-condition-diverse support protocols without inspecting query cases or
counterfactual rollout outcomes.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np


POLICIES = ("fixed", "random", "initial_condition_diversity", "posterior_action_disagreement")
DYNAMIC_POLICIES = ("posterior_action_disagreement",)
_FEATURES = ("adversary_speed_mps", "adversary_spawn_m", "sut_spawn_m")


def _features(cases: list[Mapping[str, Any]]) -> np.ndarray:
    if not cases:
        raise ValueError("support selection requires a non-empty candidate pool")
    rows = []
    for case in cases:
        try:
            row = [float(case[field]) for field in _FEATURES]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("support case lacks finite pre-execution selection fields") from exc
        if not np.isfinite(row).all():
            raise ValueError("support selection fields must be finite")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def order_support_cases(cases: list[Mapping[str, Any]], policy: str = "fixed", *, seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a deterministic ordering plus provenance for a frozen pool."""
    if policy not in POLICIES:
        raise ValueError(f"unsupported support selection policy: {policy!r}")
    if policy in DYNAMIC_POLICIES:
        raise ValueError(f"{policy} is sequential and requires support rollout posteriors")
    frozen = [dict(case) for case in cases]
    vectors = _features(frozen)
    if policy == "fixed":
        order = list(range(len(frozen)))
    elif policy == "random":
        order = np.random.default_rng(int(seed)).permutation(len(frozen)).tolist()
    else:
        # Normalize only candidate-pool metadata, then greedily maximize the
        # minimum distance to an already chosen initial condition.  Ties use
        # the frozen original order, making replay independent of platform.
        scale = np.ptp(vectors, axis=0)
        normalized = (vectors - vectors.min(axis=0)) / np.where(scale > 0.0, scale, 1.0)
        order = [0]
        remaining = set(range(1, len(frozen)))
        while remaining:
            best = max(remaining, key=lambda index: (float(np.min(np.linalg.norm(normalized[index] - normalized[order], axis=1))), -index))
            order.append(best); remaining.remove(best)
    selected = [frozen[index] for index in order]
    return selected, {
        "policy": policy, "seed": int(seed), "candidate_case_ids": [str(case["case_id"]) for case in frozen],
        "selected_case_ids": [str(case["case_id"]) for case in selected],
        "features": list(_FEATURES), "uses_query_cases": False,
        "uses_rollout_outcomes": False, "uses_hidden_rules": False,
    }
