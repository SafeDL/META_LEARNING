"""Construct immutable source-history records from an offline response bank."""

from __future__ import annotations

import hashlib
import numpy as np

from highway_sim_env.data.response_bank import ResponseBank
from replications.detour_highway_env.detour.contracts import HistoricalOutcome, ScenarioSpec


def scenario_specs(bank: ResponseBank) -> tuple[ScenarioSpec, ...]:
    """Create stable, canonical scenario identifiers for a response bank."""
    modes = np.full(len(bank.anchors), "fast_intrusion",
                    dtype="U32") if bank.modes is None else np.asarray(bank.modes, dtype=str)
    controls = getattr(bank, "scenario_controls", None)
    if controls is None:
        return tuple(
            ScenarioSpec(
                f"cutin:g={anchor[0]:.8f}:dv={anchor[1]:.8f}:mode={mode}",
                float(anchor[0]),
                float(anchor[1]),
                str(mode),
            )
            for anchor, mode in zip(bank.anchors, modes, strict=True)
        )
    controls = np.asarray(controls, dtype=float)
    return tuple(
        ScenarioSpec(
            f"cutin:g={anchor[0]:.8f}:dv={anchor[1]:.8f}:mode={mode}:"
            f"timing={control[0]:.8f}:intensity={control[1]:.8f}",
            float(anchor[0]),
            float(anchor[1]),
            str(mode),
            float(control[0]),
            float(control[1]),
        )
        for anchor, mode, control in zip(bank.anchors, modes, controls, strict=True)
    )


def source_history(bank: ResponseBank,
                   target_sut: str,
                   failure_oracle: str = "collision",
                   source_suts: tuple[str, ...] | None = None) -> tuple[HistoricalOutcome, ...]:
    """Return labels only from non-target SUTs."""
    if failure_oracle not in {"collision", "critical"}:
        raise ValueError("failure_oracle must be 'collision' or 'critical'")
    specs, target_index, records = scenario_specs(bank), bank.index_of(target_sut), []
    allowed = set(bank.sut_names) - {target_sut} if source_suts is None else set(source_suts)
    if target_sut in allowed or not allowed.issubset(bank.sut_names):
        raise ValueError("source_suts must be known and exclude the target")
    for sut_index, sut_name in enumerate(bank.sut_names):
        if sut_index == target_index or sut_name not in allowed:
            continue
        failed = bank.collisions[sut_index] if failure_oracle == "collision" else bank.collisions[
            sut_index] | bank.near_misses[sut_index]
        records.extend(
            HistoricalOutcome(f"{sut_name}:{scenario.scenario_id}", scenario, bool(failed[index]),
                              sut_name) for index, scenario in enumerate(specs))
    return tuple(records)


def within_sut_split(
    bank: ResponseBank,
    target_sut: str,
    failure_oracle: str = "collision",
    history_percent: int = 50
) -> tuple[tuple[HistoricalOutcome, ...], tuple[ScenarioSpec, ...], np.ndarray]:
    """Frozen same-SUT history/candidate split for D1 mechanism validation.

    The split is based only on canonical scenario identity, never its outcome.
    Labels for the history half are an explicit D1 input; labels for the
    candidate half remain hidden until offline scoring.
    """
    if not 0 < history_percent < 100:
        raise ValueError("history_percent must be in (0, 100)")
    specs, sut_index = scenario_specs(bank), bank.index_of(target_sut)
    history_mask = np.asarray([
        int(hashlib.sha256(item.scenario_id.encode("utf-8")).hexdigest()[:8], 16) % 100 <
        history_percent for item in specs
    ],
                              dtype=bool)
    if not history_mask.any() or history_mask.all():
        raise RuntimeError("frozen split unexpectedly produced an empty side")
    failed = bank.collisions[sut_index] if failure_oracle == "collision" else bank.collisions[
        sut_index] | bank.near_misses[sut_index]
    history = tuple(
        HistoricalOutcome(f"{target_sut}:d1-history:{spec.scenario_id}", spec, bool(failed[index]),
                          target_sut) for index, spec in enumerate(specs) if history_mask[index])
    candidate_indices = np.flatnonzero(~history_mask)
    return history, tuple(specs[index] for index in candidate_indices), candidate_indices
