"""Cut-in-only Inner training and validation task contracts."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from ..scenario.task_spec import CUTIN_LOGICAL_PARAMETER_NAMES, ScenarioMiningTaskSpec


def expand_cutin_training_domains(
    tasks: Iterable[ScenarioMiningTaskSpec], domains: Iterable[Mapping[str, Any]],
) -> list[ScenarioMiningTaskSpec]:
    """Replace each selected Cut-in train task by the requested Logical domains."""
    source_tasks = list(tasks)
    domain_rows = tuple(domains)
    if len(domain_rows) < 3:
        raise ValueError("Cut-in Inner training requires at least three Logical domains")
    if not source_tasks or any(task.functional_scenario != "cutin" for task in source_tasks):
        raise ValueError("Cut-in Inner training accepts Cut-in tasks only")
    if any(
        task.sut_split != "train"
        or task.geometry_split != "train"
        or task.logical_split != "train"
        for task in source_tasks
    ):
        raise ValueError("Cut-in Inner training may only use train-split tasks")

    expanded: list[ScenarioMiningTaskSpec] = []
    seen_ids: set[str] = set()
    for domain in domain_rows:
        domain_id = str(domain["id"])
        bounds = domain["bounds"]
        if tuple(bounds) != CUTIN_LOGICAL_PARAMETER_NAMES:
            raise ValueError("Cut-in Logical domains must use the canonical parameter order")
        normalized_bounds = {
            name: tuple(float(value) for value in bounds[name])
            for name in CUTIN_LOGICAL_PARAMETER_NAMES
        }
        for task in source_tasks:
            task_id = f"{task.task_id}:{domain_id}"
            if task_id in seen_ids:
                raise ValueError("duplicate Cut-in Inner task id")
            seen_ids.add(task_id)
            expanded.append(replace(
                task,
                task_id=task_id,
                logical_domain_id=domain_id,
                logical_domain_bounds=normalized_bounds,
            ))
    for task in expanded:
        task.validate()
    return expanded


def select_cutin_validation_tasks(
    tasks: Iterable[ScenarioMiningTaskSpec], geometry_ids: Iterable[str] = (),
) -> list[ScenarioMiningTaskSpec]:
    """Return only the retained-topology validation SUT × Logical-domain slice."""
    allowed_geometries = frozenset(str(value) for value in geometry_ids)
    selected = [
        task for task in tasks
        if task.functional_scenario == "cutin"
        and task.functional_split == "train"
        and task.sut_split == "validation"
        and task.geometry_split == "validation"
        and task.logical_split == "validation"
        and (not allowed_geometries or task.geometry_id in allowed_geometries)
    ]
    if not selected:
        raise ValueError("Cut-in Inner validation selection is empty")
    if any("test" in (task.sut_split, task.geometry_split, task.logical_split) for task in selected):
        raise ValueError("Cut-in Inner validation must not access a test split")
    return selected
