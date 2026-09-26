"""Pure-data predicates shared by the measured-bank replay paths."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def is_usable_outcome(row: Mapping[str, Any]) -> bool:
    """Whether a normalized, observed outcome can train the failure model."""
    if row.get("inconclusive", False):
        return False
    collision = row.get("ego_collision")
    if collision is True:
        return True
    return collision is False and row.get("completed") is True


def is_parent_pass(row: Mapping[str, Any]) -> bool:
    """A regression candidate must have a complete, collision-free parent run."""
    return (not row.get("inconclusive", False)
            and row.get("ego_collision") is False
            and row.get("completed") is True)


def contextual_history(history: list[dict], candidates: list[dict]) -> list[dict]:
    contexts: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        scenario = item.get("scenario", {})
        context = item.get("context_id") or scenario.get("context_id", "legacy_unspecified")
        contexts[item["template_id"]].add(context)
    if any(len(values) > 1 for values in contexts.values()):
        raise ValueError(f"a selector task contains multiple contexts: {contexts}")
    allowed = {template: next(iter(values)) for template, values in contexts.items()}
    return [row for row in history
            if row.get("visibility") != "evaluator_only"
            and row.get("template_id") in allowed
            and row.get("context_id", row.get("scenario", {}).get("context_id",
                                                                  "legacy_unspecified"))
            == allowed[row["template_id"]]]


def task_reward(mode: str, outcome: Mapping[str, Any],
                parent_pass: bool | None = None) -> int:
    if mode not in {"regression", "cross_agent"}:
        raise ValueError(f"unsupported task mode: {mode}")
    if mode == "regression" and not isinstance(parent_pass, bool):
        raise ValueError("regression tasks require an explicit parent-pass label")
    collision = (not outcome.get("inconclusive", False)
                 and outcome.get("ego_collision") is True)
    if mode == "cross_agent":
        return int(collision)
    return int(parent_pass and collision)


def history_kind(records: list[dict]) -> str:
    failures = any(row.get("ego_collision") is True and is_usable_outcome(row)
                   for row in records)
    passes = any(is_parent_pass(row) for row in records)
    if failures and passes:
        return "failure_and_pass"
    if failures:
        return "failure_only"
    if passes:
        return "pass_only"
    return "empty"


def excluded_history_counts(records: list[dict], candidates: list[dict]) -> dict[str, int]:
    allowed_contexts: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        scenario = item.get("scenario", {})
        allowed_contexts[item["template_id"]].add(
            item.get("context_id") or scenario.get("context_id", "legacy_unspecified"))
    excluded = defaultdict(int)
    for row in records:
        if row.get("visibility") == "evaluator_only":
            excluded["evaluator_only"] += 1
        elif row.get("template_id") not in allowed_contexts:
            excluded["template_not_in_candidate_context"] += 1
        elif row.get("context_id", row.get("scenario", {}).get("context_id",
                                                               "legacy_unspecified")) not in \
                allowed_contexts[row["template_id"]]:
            excluded["incompatible_context"] += 1
        elif not is_usable_outcome(row):
            excluded["inconclusive_or_incomplete"] += 1
    return dict(excluded)
