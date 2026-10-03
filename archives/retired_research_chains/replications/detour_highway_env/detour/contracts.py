"""Data contracts that keep a selector independent of target outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    """An executable Cut-in scenario, identified independently of its array row."""

    scenario_id: str
    initial_gap: float
    relative_speed: float
    mode: str
    timing: float | None = None
    intensity: float | None = None


@dataclass(frozen=True)
class HistoricalOutcome:
    """A labelled execution from a source SUT; never a target-truth record."""

    execution_id: str
    scenario: ScenarioSpec
    failed: bool
    source_sut: str
