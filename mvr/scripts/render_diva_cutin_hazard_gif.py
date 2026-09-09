"""Render one verified hazardous design from the current Cut-in space."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from ..diva.behavior import DivaCutInBehavior
from ..diva.types import DivaCutInDesign
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.task_spec import logical_parameter_names
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner
from .render_cutin_inner_policy_gif import (
    ADVERSARY_COLOR,
    SUT_COLOR,
    VISUAL_ENVIRONMENT_OVERRIDES,
    _paint_role,
)
from .render_diva_source_replay_gif import _frame, _save


HAZARD_SUT_REF = "idm_normal"
HAZARD_DESIGN = DivaCutInDesign(0, (-1.0, -1.0 / 6.0, -1.0, -1.0, -0.4))
HAZARD_SEED = 9200


def run(output_path: str, report_path: str) -> dict[str, Any]:
    base_task = next(
        task for task in load_taskbook("mvr/configs/taskbook.json")
        if task.sut_ref == HAZARD_SUT_REF
        and task.geometry_id == "cutin-g01"
        and task.functional_scenario == "cutin"
    )
    task = replace(
        base_task,
        task_id="cutin-g01-idm-normal-hazard-replay",
        logical_domain_id="cutin_physical_probe",
        logical_domain_bounds={name: (-1.0, 1.0) for name in logical_parameter_names("cutin")},
    )
    task.validate()
    values = mvr_parameter_spaces()["cutin"].decode(HAZARD_DESIGN.scenario_action())
    episode = ScenarioExecutor(load_adapters(), mvr_parameter_spaces()).reset(
        task,
        HAZARD_DESIGN.scenario_action(),
        episode_seed=HAZARD_SEED,
        environment_overrides=VISUAL_ENVIRONMENT_OVERRIDES | {"horizon": 480},
    )
    behavior = DivaCutInBehavior()
    frames: list[Any] = []
    label = (
        "verified collision | initial gap={:.1f} m | ego={:.1f} red={:.1f} m/s | "
        "start={:.1f} m | path={:.1f} m"
    ).format(
        float(values["initial_gap_m"]),
        float(values["ego_initial_speed_mps"]),
        float(values["ego_initial_speed_mps"]) + float(values["relative_speed_mps"]),
        float(values["cutin_start_offset_m"]),
        float(values["cutin_path_length_m"]),
    )

    def capture(current: Any, step: int, info: Mapping[str, Any]) -> None:
        if step == 0:
            current.env.engine.main_camera.track(current.sut)
            _paint_role(current.sut, SUT_COLOR)
            _paint_role(current.adversary, ADVERSARY_COLOR)
        elif step % 2 == 0:
            frames.append(_frame(current, info, step, label))

    try:
        rollout = HierarchicalRunner(480).rollout(
            episode, "cutin", step_action=behavior, step_callback=capture
        )
    finally:
        episode.env.close()
    _save(frames, Path(output_path))
    report = {
        "scope": "current Cut-in physical space hazard replay",
        "sut_ref": HAZARD_SUT_REF,
        "design": HAZARD_DESIGN.to_dict(),
        "physical_parameters": values,
        "episode_seed": HAZARD_SEED,
        "score": float(rollout.outcome.get("valid_target_collision", False)),
        "outcome": rollout.outcome,
        "frames": len(frames),
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
