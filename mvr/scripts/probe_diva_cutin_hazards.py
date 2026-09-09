"""Probe whether physically admissible Cut-in designs can produce hazards."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from ..diva.episode_executor import DivaEpisodeExecutor
from ..diva.types import DivaCutInDesign
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.task_spec import logical_parameter_names
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner


# Low initial gaps, an early legal onset, and a slower red vehicle are the
# most direct physically admissible ways to challenge the following SUT.
PROBE_DESIGNS = (
    DivaCutInDesign(0, (-1.0, -1.0 / 6.0, -1.0, -1.0, -0.4)),
    DivaCutInDesign(1, (-1.0, -1.0 / 6.0, -1.0, -1.0, -0.4)),
    DivaCutInDesign(0, (-1.0, 0.0, -1.0, -1.0, -0.3)),
    DivaCutInDesign(1, (-1.0, 0.0, -1.0, -1.0, -0.3)),
    DivaCutInDesign(0, (-0.8, -0.2, -0.8, -1.0, -0.5)),
    DivaCutInDesign(1, (-0.8, -0.2, -0.8, -1.0, -0.5)),
)
PROBE_SUT_REFS = ("idm_cautious", "idm_normal", "idm_assertive", "idm_fast_small_gap")


def run(output_path: str) -> dict[str, Any]:
    taskbook = load_taskbook("mvr/configs/taskbook.json")
    spaces = mvr_parameter_spaces()
    executor = DivaEpisodeExecutor(
        ScenarioExecutor(load_adapters(), spaces),
        HierarchicalRunner(max_steps=480),
        environment_horizon=480,
    )
    rows = []
    for sut_index, sut_ref in enumerate(PROBE_SUT_REFS):
        base_task = next(
            task for task in taskbook
            if task.sut_ref == sut_ref
            and task.geometry_id == "cutin-g01"
            and task.functional_scenario == "cutin"
        )
        full_task = replace(
            base_task,
            task_id=f"cutin-g01-{sut_ref}-physical-risk-probe",
            logical_domain_id="cutin_physical_probe",
            logical_domain_bounds={name: (-1.0, 1.0) for name in logical_parameter_names("cutin")},
        )
        full_task.validate()
        for design_index, design in enumerate(PROBE_DESIGNS):
            observation = executor.run(
                full_task, design, episode_seed=9100 + 100 * sut_index + design_index
            )
            outcome = observation.outcome
            rows.append({
                "sut_ref": sut_ref,
                "design": design.to_dict(),
                "physical_parameters": spaces["cutin"].decode(design.scenario_action()),
                "score": observation.score,
                "status": observation.status,
                "is_valid_episode": observation.is_valid_episode,
                "cutin_actual_onset": outcome["cutin_actual_onset"],
                "maneuver_completed": outcome["maneuver_completed"],
                "termination_reason": outcome["termination_reason"],
            })
    report = {
        "scope": "current physically admissible global Cut-in space",
        "probe_design_count": len(rows),
        "hazard_found": any(row["score"] > 0.0 for row in rows),
        "rows": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
