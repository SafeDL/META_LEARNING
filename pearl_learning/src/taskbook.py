"""Frozen logical-scenario taskbooks; no train/eval-time generation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

from .io import content_hash, write_json
from .routes import route_hash_payload
from .task_spec import LogicalScenarioTaskSpec, TASK_SCHEMA


TASKBOOK_SCHEMA = "logical_merge_taskbook"
SPLITS = ("meta_train", "meta_validation", "meta_test_template", "meta_test_logical")


def load_geometry_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = config.get("geometry_catalog")
    if not isinstance(data, Mapping) or data.get("schema") != "logical_merge_geometry_catalog":
        raise ValueError("geometry_catalog must use the logical_merge_geometry_catalog schema")
    geometries = data.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("geometry catalog must contain a non-empty geometries list")
    ids = [str(item.get("geometry_id", "")) for item in geometries if isinstance(item, Mapping)]
    if len(ids) != len(geometries) or len(ids) != len(set(ids)):
        raise ValueError("geometry catalog has missing or duplicate geometry_id values")
    rules = data.get("target_contact_rules")
    if not isinstance(rules, Mapping) or set(rules) != set(ids):
        raise ValueError("geometry catalog must define exactly one target_contact_rules entry per geometry")
    allow_hidden_rules = bool(config.get("task_definition", {}).get("allow_hidden_reward_rule_variants", True))
    train_variants = data.get("meta_train_target_contact_rule_variants", {}) if allow_hidden_rules else {}
    evaluation_variants = data.get("evaluation_target_contact_rule_variants", {}) if allow_hidden_rules else {}
    if not isinstance(train_variants, Mapping) or not set(train_variants) <= set(ids):
        raise ValueError("meta_train_target_contact_rule_variants must map known geometry ids to rule lists")
    if not isinstance(evaluation_variants, Mapping) or not set(evaluation_variants) <= set(ids):
        raise ValueError("evaluation_target_contact_rule_variants must map known geometry ids to rule lists")
    if set(train_variants) & set(evaluation_variants):
        raise ValueError("a geometry cannot have both training and evaluation rule variants")
    entry_orders = data.get("target_contact_entry_orders", {})
    if not isinstance(entry_orders, Mapping) or set(entry_orders) != set(ids):
        raise ValueError("geometry catalog must define exactly one target_contact_entry_orders entry per geometry")
    entry_semantics = data.get("target_contact_entry_order_semantics")
    if entry_semantics != "pre_step_arrival_time":
        raise ValueError("geometry catalog must freeze pre_step_arrival_time entry-order semantics")
    resolved: list[dict[str, Any]] = []
    spawn_override = config.get("case_sampling", {}).get("spawn_regions_override")
    if spawn_override is not None:
        if not isinstance(spawn_override, Mapping) or set(spawn_override) != {"adversary", "sut"}:
            raise ValueError("case_sampling.spawn_regions_override must define adversary and sut")
    for item in geometries:
        source = dict(item); source_id = str(source["geometry_id"])
        split = source.get("split")
        if split == "meta_train" and source_id in evaluation_variants:
            raise ValueError("evaluation rule variants are not permitted for meta_train geometries")
        if split != "meta_train" and source_id in train_variants:
            raise ValueError("rule variants are permitted only for meta_train geometries")
        variants = train_variants if split == "meta_train" else evaluation_variants
        candidate_rules = variants.get(source_id, [rules[source_id]])
        if not isinstance(candidate_rules, list) or not candidate_rules:
            raise ValueError(f"rule variants for {source_id} must be a non-empty list")
        for rule in candidate_rules:
            if not isinstance(rule, Mapping):
                raise ValueError(f"target-contact rule for {source_id} must be a mapping")
            geometry = dict(source)
            if spawn_override is not None:
                geometry["spawn_regions"] = {
                    role: [float(value) for value in spawn_override[role]]
                    for role in ("adversary", "sut")
                }
            priority = dict(geometry.get("priority_spec", {}))
            if allow_hidden_rules:
                priority["target_contact_entry_order"] = str(entry_orders[source_id])
                priority["target_contact_entry_order_semantics"] = str(entry_semantics)
                priority.update(dict(rule))
            else:
                priority["target_contact_entry_order"] = "any"
                priority["target_contact_speed_relation"] = "any"
                priority["target_contact_speed_margin_mps"] = 0.0
            relation = str(priority.get("target_contact_entry_order", "any"))
            if relation == "any":
                relation = str(priority.get("target_contact_speed_relation", "any"))
            if len(candidate_rules) > 1:
                geometry["geometry_id"] = f"{source_id}__rule_{relation}"
            geometry["priority_spec"] = priority
            resolved.append(geometry)
    expanded_ids = [str(item["geometry_id"]) for item in resolved]
    if len(expanded_ids) != len(set(expanded_ids)):
        raise ValueError("rule expansion produced duplicate geometry ids")
    return resolved


def _global_case_seed(seed: int, geometry_id: str) -> int:
    return int(content_hash({"taskbook_seed": int(seed), "geometry_id": geometry_id})[:8], 16) % (2**31 - 2) + 1


def _task_from_geometry(geometry: Mapping[str, Any], seed: int) -> LogicalScenarioTaskSpec:
    required = {
        "geometry_id", "split", "family", "logical_type", "map_config", "adversary_route", "sut_route",
        "spawn_regions", "priority_spec", "conflict_spec",
    }
    missing = required - set(geometry)
    if missing:
        raise ValueError(f"geometry {geometry.get('geometry_id')} misses fields {sorted(missing)}")
    geometry_id = str(geometry["geometry_id"])
    adversary_route, sut_route, conflict = (
        dict(geometry["adversary_route"]), dict(geometry["sut_route"]), dict(geometry["conflict_spec"])
    )
    # A taskbook becomes executable only after build_taskbook.py resolves these
    # recipe fingerprints to real-map fingerprints.  They remain deterministic
    # and globally distinct before that operation.
    return LogicalScenarioTaskSpec(
        schema=TASK_SCHEMA,
        task_id=f"{geometry['split']}_{geometry_id}",
        split=str(geometry["split"]),
        family=str(geometry["family"]),
        logical_type=str(geometry["logical_type"]),
        geometry_id=geometry_id,
        map_config=dict(geometry["map_config"]),
        adversary_route=adversary_route,
        sut_route=sut_route,
        spawn_regions={key: [float(x) for x in value] for key, value in dict(geometry["spawn_regions"]).items()},
        priority_spec=dict(geometry["priority_spec"]),
        conflict_spec=conflict,
        case_seed=_global_case_seed(seed, geometry_id),
        map_hash=content_hash({"geometry_id": geometry_id, "map_config": geometry["map_config"]}),
        adversary_route_hash=content_hash({"geometry_id": geometry_id, "map_config": geometry["map_config"], "route": route_hash_payload(adversary_route)}),
        sut_route_hash=content_hash({"geometry_id": geometry_id, "map_config": geometry["map_config"], "route": route_hash_payload(sut_route)}),
        conflict_hash=content_hash(conflict),
    )


def build_taskbook(config: Mapping[str, Any]) -> dict[str, list[LogicalScenarioTaskSpec]]:
    """Create a deterministic candidate taskbook from the versioned catalog.

    This helper is only for ``build_taskbook.py`` and tests.  Train/evaluate
    commands deliberately require :func:`load_taskbook` instead.
    """
    seed = int(config.get("experiment", {}).get("taskbook_seed", 7301))
    result = {split: [] for split in SPLITS}
    for geometry in load_geometry_catalog(config):
        task = _task_from_geometry(geometry, seed)
        if task.split not in result:
            raise ValueError(f"unknown task split {task.split!r}")
        result[task.split].append(task)
    validate_taskbook(result)
    return result


def taskbook_payload(taskbook: Mapping[str, Iterable[LogicalScenarioTaskSpec]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [task.to_dict() for task in taskbook.get(split, [])] for split in SPLITS}


def validate_taskbook(taskbook: Mapping[str, Iterable[LogicalScenarioTaskSpec]]) -> None:
    seen_ids: set[str] = set()
    split_values: dict[str, set[str]] = {name: set() for name in SPLITS}
    for split in SPLITS:
        tasks = list(taskbook.get(split, []))
        if not tasks:
            raise ValueError(f"{split} must contain at least one physical geometry")
        for task in tasks:
            task.validate()
            if task.task_id in seen_ids:
                raise ValueError(f"duplicate task id: {task.task_id}")
            if task.split != split:
                raise ValueError(f"task {task.task_id} is in wrong split")
            seen_ids.add(task.task_id)
            split_values[split].add(task.geometry_id)
    for field in ("geometry_id", "map_hash", "adversary_route_hash", "sut_route_hash"):
        sets = [
            {str(getattr(task, field)) for task in taskbook.get(split, [])}
            for split in SPLITS
        ]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                if sets[left] & sets[right]:
                    raise ValueError(f"{field} leaks between {SPLITS[left]} and {SPLITS[right]}")
    train_types = {task.logical_type for task in taskbook["meta_train"]}
    held_out = {task.logical_type for task in taskbook["meta_test_logical"]}
    if held_out & train_types:
        raise ValueError("held-out logical type leaked into meta-train")


def replace_geometry_hashes(task: LogicalScenarioTaskSpec, *, map_hash: str, adversary_route_hash: str | None = None,
                            sut_route_hash: str | None = None, conflict_hash: str | None = None) -> LogicalScenarioTaskSpec:
    """Return a task with hashes resolved from an instantiated map."""
    updated = replace(
        task,
        map_hash=map_hash,
        adversary_route_hash=adversary_route_hash or task.adversary_route_hash,
        sut_route_hash=sut_route_hash or task.sut_route_hash,
        conflict_hash=conflict_hash or task.conflict_hash,
    )
    updated.validate()
    return updated


def save_taskbook(taskbook: Mapping[str, Iterable[LogicalScenarioTaskSpec]], output_dir: str | Path) -> str:
    validate_taskbook(taskbook)
    payload = taskbook_payload(taskbook)
    digest = content_hash(payload)
    root = Path(output_dir)
    for split, tasks in payload.items():
        write_json(root / f"{split}_tasks.json", tasks)
    return digest


def load_taskbook(path: str | Path) -> dict[str, list[LogicalScenarioTaskSpec]]:
    root = Path(path)
    metadata = json.loads((root / "taskbook_provenance.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != TASKBOOK_SCHEMA or metadata.get("task_schema") != TASK_SCHEMA:
        raise ValueError("taskbook schema is unsupported or not executable")
    taskbook: dict[str, list[LogicalScenarioTaskSpec]] = {}
    for split in SPLITS:
        data = json.loads((root / f"{split}_tasks.json").read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{split} taskbook payload is not a list")
        taskbook[split] = [LogicalScenarioTaskSpec.from_dict(row) for row in data]
    validate_taskbook(taskbook)
    if content_hash(taskbook_payload(taskbook)) != metadata.get("taskbook_hash"):
        raise ValueError("taskbook SHA-256 does not match its payload")
    return taskbook
