from __future__ import annotations

from pathlib import Path

import yaml

from mvr.experiments.cutin_inner import (
    expand_cutin_training_domains,
    select_cutin_validation_tasks,
)
from mvr.scenario.registry import load_geometry_catalog
from mvr.scripts.evaluate_cutin_inner_validation import _support_effects
from mvr.scenario.task_spec import CUTIN_LOGICAL_PARAMETER_NAMES, ScenarioMiningTaskSpec
from mvr.scenario.taskbook import load_taskbook


def _task(
    *,
    sut_split: str = "train",
    geometry_split: str = "train",
    logical_split: str = "train",
) -> ScenarioMiningTaskSpec:
    return ScenarioMiningTaskSpec(
        task_id=f"cutin-{sut_split}-{logical_split}",
        sut_ref="idm",
        functional_scenario="cutin",
        geometry_id="cutin-g01",
        geometry_hash="a" * 64,
        geometry_seed=201,
        adapter_id="cutin",
        interaction_schema_id="two_route_conflict",
        sut_split=sut_split,
        geometry_split=geometry_split,
        logical_domain_id="source",
        logical_domain_bounds={name: (-0.25, 0.25) for name in CUTIN_LOGICAL_PARAMETER_NAMES},
        logical_parameter_mask=(True,) * len(CUTIN_LOGICAL_PARAMETER_NAMES),
        logical_split=logical_split,
    )


def test_cutin_training_expansion_constructs_three_train_domains() -> None:
    domains = [
        {"id": name, "bounds": {key: bounds for key in CUTIN_LOGICAL_PARAMETER_NAMES}}
        for name, bounds in (("early", (-0.25, -0.10)), ("middle", (-0.08, 0.08)), ("late", (0.10, 0.25)))
    ]
    expanded = expand_cutin_training_domains([_task()], domains)
    assert {task.logical_domain_id for task in expanded} == {"early", "middle", "late"}
    assert all(task.sut_split == task.geometry_split == task.logical_split == "train" for task in expanded)


def test_cutin_training_expansion_accepts_one_domain() -> None:
    bounds = {key: (-0.25, 0.25) for key in CUTIN_LOGICAL_PARAMETER_NAMES}
    expanded = expand_cutin_training_domains(
        [_task()], [{"id": "balanced", "bounds": bounds}]
    )

    assert len(expanded) == 1
    assert expanded[0].logical_domain_id == "balanced"


def test_cutin_inner_config_selects_one_training_task() -> None:
    config = yaml.safe_load(
        Path("mvr/configs/cutin_inner.yaml").read_text(encoding="utf-8")
    )
    settings = config["cutin_inner"]
    prior = config["interaction_prior"]
    source = [
        task
        for task in load_taskbook(config["taskbook"])
        if task.functional_scenario == "cutin"
        and task.sut_split == task.geometry_split == task.logical_split == "train"
        and task.sut_ref in settings["training_sut_refs"]
        and task.geometry_id in settings["training_geometry_ids"]
    ]
    expanded = expand_cutin_training_domains(
        source, settings["training_logical_domains"]
    )

    assert len(expanded) == 1
    assert expanded[0].sut_ref == "idm_normal"
    assert expanded[0].geometry_id == "cutin-g01"
    assert expanded[0].logical_domain_id == "balanced_interaction"
    assert prior["episodes_per_task"] == 40
    assert prior["warmup_episodes"] == 5
    assert prior["event_sample_fraction"] == 0.25
    assert prior["event_action_weight"] == 0.0
    assert prior["gamma"] == 0.99
    assert config["context_meta"]["gamma"] == 0.99
    assert config["model"]["state_dim"] == 30
    assert settings["freeze_static_representation_during_interaction_prior"] is True


def test_all_cutin_geometries_allow_the_completion_budget() -> None:
    geometries = load_geometry_catalog()

    assert all(
        geometries[f"cutin-g{index:02d}"].horizon == 360
        for index in range(1, 6)
    )


def test_cutin_validation_selector_keeps_validation_sut_and_domain_only() -> None:
    validation = _task(
        sut_split="validation",
        geometry_split="validation",
        logical_split="validation",
    )
    selected = select_cutin_validation_tasks([_task(), validation])
    assert selected == [validation]


def test_support_effects_report_latent_and_planner_action_changes() -> None:
    records = []
    for shots, z, action in (
        (0, [0.0, 0.0], [0.0] * 4),
        (1, [0.1, 0.0], [0.2, 0.0, 0.0, 0.0]),
    ):
        records.append({
            "policy": "adapted_h_z", "support_shots": shots, "task_id": "task",
            "seed": 11, "query_id": "query", "z": z,
            "mean_planner_action": action,
            "mean_executed_vehicle_action": action[-2:],
        })
    report = _support_effects(records, (0, 1))
    assert report["1"]["z_changed"] is True
    assert report["1"]["planner_action_changed"] is True
