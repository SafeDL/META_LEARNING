"""Executable static-contract audit required alongside the topology audit."""
from __future__ import annotations

import argparse
import inspect

from pearl_learning.src.casebook import CASEBOOK_SCHEMA, load_casebook, validate_casebook_disjoint
from pearl_learning.src.evaluator import evaluate_fewshot
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.observation import observation_dim, observation_fields
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.replay import ReplayEpisode
from pearl_learning.src.task_spec import TASK_SCHEMA
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload, validate_taskbook


def _split_hashes_are_disjoint(taskbook) -> bool:
    fields = ("geometry_id", "map_hash", "adversary_route_hash", "sut_route_hash")
    splits = list(taskbook)
    for field in fields:
        values = [{getattr(task, field) for task in taskbook[split]} for split in splits]
        if any(values[left] & values[right] for left in range(len(values)) for right in range(left + 1, len(values))):
            return False
    return True


def _audit_taskbook(config, taskbook):
    """Restrict a pilot audit to the exact task IDs used by that pilot."""
    requested = config.get("method_flow_pilot", {}).get("task_ids")
    if not requested:
        return taskbook
    selected = {}
    for split, geometry_ids in requested.items():
        wanted = set(map(str, geometry_ids))
        tasks = [task for task in taskbook.get(split, []) if task.geometry_id in wanted]
        if {task.geometry_id for task in tasks} != wanted:
            raise ValueError(f"method-flow pilot {split} task ids do not match frozen taskbook")
        selected[split] = tasks
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_config(args.config); taskbook = load_taskbook(args.taskbook); validate_taskbook(taskbook)
    audited_taskbook = _audit_taskbook(cfg, taskbook)
    all_tasks = [task for tasks in audited_taskbook.values() for task in tasks]
    v2_metric = str(cfg.get("critical_metric", {}).get("schema")) == "spatiotemporal_near_miss_v2"
    required_casebook_schema = CASEBOOK_SCHEMA if v2_metric else None
    casebooks = {
        task.task_id: load_casebook(
            task, args.casebook_root, required_schema=required_casebook_schema,
        )
        for task in all_tasks
    }
    try:
        validate_casebook_disjoint(casebooks); casebooks_are_disjoint = True
    except ValueError:
        casebooks_are_disjoint = False
    agent_update = inspect.getsource(PEARLAgent.update)
    agent_hash = inspect.getsource(PEARLAgent.parameter_hash) + inspect.getsource(PEARLAgent.module_hashes)
    evaluator_source = inspect.getsource(evaluate_fewshot)
    required_episode_fields = {"task_id", "episode_id", "case_id", "collection_mode", "posterior_version", "terminated", "truncated", "termination_reason"}
    payload_hash = content_hash(taskbook_payload(taskbook))
    checks = {
        "schema_current": all(task.schema == TASK_SCHEMA for task in all_tasks),
        "taskbook_hash_verified": bool(payload_hash),
        "split_route_hash_isolation": _split_hashes_are_disjoint(audited_taskbook),
        "casebook_split_isolation": casebooks_are_disjoint,
        "observation_contract": int(cfg["environment"]["observation_dim"]) == observation_dim(cfg["environment"]["observation_schema"]),
        "observation_label_free": (
            "template_index" not in observation_fields(cfg["environment"]["observation_schema"])
            and "adversary_priority" not in observation_fields(cfg["environment"]["observation_schema"])
            and "sut_priority" not in observation_fields(cfg["environment"]["observation_schema"])
        ),
        "truncation_bootstrap": (
            "transition.terminated" in agent_update
            and "transition.truncated" not in agent_update
            and "(1 - done)" in agent_update
        ),
        "episode_replay_provenance": required_episode_fields <= set(ReplayEpisode.__dataclass_fields__),
        "no_gradient_hash_scope": (
            all(token in agent_hash for token in ("context_encoder", "actor", "q1", "q2", "target_q1", "target_q2", "log_alpha"))
            and "module_hashes_before" in evaluator_source
            and "module_hashes_after" in evaluator_source
            and "if before != after or module_hashes_before != module_hashes_after" in evaluator_source
        ),
    }
    result = {
        "schema": "logical_merge_integrity_audit",
        "taskbook_hash": payload_hash,
        "audited_task_ids": sorted(casebooks),
        "required_casebook_schema": required_casebook_schema,
        "casebook_hashes": {task_id: content_hash(book) for task_id, book in casebooks.items()},
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    write_json(args.output, result)
    if result["status"] != "pass":
        raise SystemExit("integrity audit failed")


if __name__ == "__main__":
    main()
