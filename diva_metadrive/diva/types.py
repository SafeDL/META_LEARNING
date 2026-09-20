"""Immutable DIVA contracts independent of the simulator and SUT identity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

from ..provenance import content_hash
from ..scenario.parameter_space import NormalizedScenarioAction

DIVA_SCHEMA = "diva_mine_cutin_constant_speed_physical_v2"
ObservationStatus = Literal["valid_event", "completed_noncritical", "censored", "invalid", ]


@dataclass(frozen=True)
class DivaCutInDesign:
    """One normalized five-parameter Cut-in scenario, with no SUT input."""

    candidate_index: int
    logical_continuous: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = self.logical_continuous
        if self.candidate_index not in (0, 1):
            raise ValueError("DIVA Cut-in candidate_index must be 0 or 1")
        if not np.isfinite(values).all() or any(abs(float(value)) > 1.0 for value in values):
            raise ValueError("DIVA design controls must be finite normalized values")

    def scenario_action(self) -> NormalizedScenarioAction:
        return NormalizedScenarioAction(self.candidate_index, self.logical_continuous)

    def feature_vector(self) -> np.ndarray:
        """Continuous [0, 1]^5 scenario vector; candidate partitions the GP."""
        values = np.asarray(self.logical_continuous, dtype=np.float64)
        return (values + 1.0) * 0.5

    @property
    def design_id(self) -> str:
        return content_hash({"schema": DIVA_SCHEMA, **asdict(self)})

    def to_dict(self) -> dict[str, Any]:
        return {"schema": DIVA_SCHEMA, "design_id": self.design_id, **asdict(self)}


@dataclass(frozen=True)
class DivaObservation:
    """One auditable call: formal score and learning response remain separate."""

    design: DivaCutInDesign
    task_id: str
    sut_ref: str
    geometry_id: str
    logical_domain_id: str
    episode_seed: int
    score: float
    is_valid_episode: bool
    status: ObservationStatus
    posterior_eligible: bool
    vulnerability_response: float | None
    outcome: Mapping[str, Any]
    concrete_scenario: Mapping[str, Any]
    behavior_contract_hash: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if self.status not in {"valid_event", "completed_noncritical", "censored", "invalid"}:
            raise ValueError("unknown DIVA observation status")
        if float(self.score) not in (0.0, 0.5, 1.0):
            raise ValueError("DIVA score must use the formal valid-critical scale")
        if self.posterior_eligible != (self.status in {"valid_event", "completed_noncritical"}):
            raise ValueError("posterior eligibility must agree with observation status")
        if self.posterior_eligible:
            if self.vulnerability_response is None or not np.isfinite(self.vulnerability_response):
                raise ValueError(
                    "eligible DIVA observation requires a finite vulnerability response")
            if not 0.0 <= float(self.vulnerability_response) <= 1.0:
                raise ValueError("vulnerability response must lie in [0, 1]")
        elif self.vulnerability_response is not None:
            raise ValueError("ineligible DIVA observation must not carry a vulnerability response")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = DIVA_SCHEMA
        payload["design"] = self.design.to_dict()
        return payload
