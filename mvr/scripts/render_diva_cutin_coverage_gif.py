"""Render display-only replays spanning the frozen DIVA source domain."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..diva.behavior import DivaCutInBehavior
from ..diva.types import DIVA_SCHEMA, DivaCutInDesign
from ..evaluation.fewshot_inner import valid_critical_score
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner
from .render_cutin_inner_policy_gif import (
    ADVERSARY_COLOR,
    SUT_COLOR,
    VISUAL_ENVIRONMENT_OVERRIDES,
    _paint_role,
)
from .render_diva_source_replay_gif import _frame, _save


REPRESENTATIVE_DESIGNS = (
    DivaCutInDesign(0, (-0.15, -0.50, -0.60, -0.95, 0.00)),
    DivaCutInDesign(0, (0.15, 0.50, 0.60, -0.85, 0.40)),
    DivaCutInDesign(1, (-0.15, 0.50, -0.60, -0.85, 0.10)),
    DivaCutInDesign(1, (0.15, -0.50, 0.60, -0.95, 0.20)),
)


def _label(index: int, values: Mapping[str, float | str]) -> str:
    return (
        "coverage {}/4 | {} | ego={:.1f} red={:.1f} m/s | "
        "initial gap={:.1f} m | path={:.1f} m | start={:.1f} m"
    ).format(
        index + 1,
        values["route_or_conflict_candidate"],
        float(values["ego_initial_speed_mps"]),
        float(values["ego_initial_speed_mps"]) + float(values["relative_speed_mps"]),
        float(values["initial_gap_m"]),
        float(values["cutin_path_length_m"]),
        float(values["cutin_start_offset_m"]),
    )


def run(output_path: str, report_path: str) -> dict[str, Any]:
    task = next(
        row for row in load_taskbook("mvr/configs/taskbook.json")
        if row.task_id == "cutin-g01-cautious-cutin_interaction_core"
    )
    executor = ScenarioExecutor(load_adapters(), mvr_parameter_spaces())
    frames: list[Any] = []
    records: list[dict[str, Any]] = []
    for index, design in enumerate(REPRESENTATIVE_DESIGNS):
        values = mvr_parameter_spaces()["cutin"].decode(design.scenario_action())
        label = _label(index, values)
        episode = executor.reset(
            task,
            design.scenario_action(),
            episode_seed=1701 + index,
            environment_overrides=VISUAL_ENVIRONMENT_OVERRIDES | {"horizon": 300},
        )
        behavior = DivaCutInBehavior()

        def capture(current: Any, step: int, info: Mapping[str, Any]) -> None:
            if step == 0:
                current.env.engine.main_camera.track(current.sut)
                _paint_role(current.sut, SUT_COLOR)
                _paint_role(current.adversary, ADVERSARY_COLOR)
            elif step % 3 == 0:
                frames.append(_frame(current, info, step, label))

        try:
            rollout = HierarchicalRunner(300).rollout(
                episode,
                "cutin",
                step_action=behavior,
                step_callback=capture,
            )
        finally:
            episode.env.close()
        records.append(
            {
                "design": design.to_dict(),
                "physical_parameters": values,
                "red_initial_speed_mps": behavior.prescribed_speed_mps,
                "score": valid_critical_score(rollout.outcome),
                "maneuver_completed": any(
                    bool(row["info"].get("semantic_maneuver_completed", False))
                    for row in rollout.transitions
                ),
                "termination_reason": rollout.outcome.get("termination_reason"),
            }
        )
    _save(frames, Path(output_path))
    report = {
        "schema": "diva_display_coverage",
        "design_schema": DIVA_SCHEMA,
        "task_id": task.task_id,
        "display_only": True,
        "excluded_from_source_bank": True,
        "excluded_from_experiment_budget": True,
        "records": records,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    run(args.output, args.report)


if __name__ == "__main__":
    main()
