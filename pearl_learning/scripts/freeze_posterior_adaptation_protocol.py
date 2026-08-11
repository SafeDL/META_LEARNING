"""Freeze posterior-adaptation hypotheses before holdout evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = read_config(args.config)
    protocol = config.get("posterior_adaptation")
    if not isinstance(protocol, dict) or protocol.get("schema") != "posterior_adaptation_protocol_v1":
        raise ValueError("configuration lacks a posterior_adaptation_protocol_v1 declaration")
    taskbook = load_taskbook(args.taskbook)
    shots = [int(value) for value in config["evaluation"]["shots"]]
    if shots != [0, 1, 2, 4, 8]:
        raise ValueError("posterior adaptation requires frozen shots [0, 1, 2, 4, 8]")
    if config["evaluation"].get("context_protocol") != "fixed_nested_v1":
        raise ValueError("posterior adaptation requires fixed_nested_v1 context sampling")
    if int(config["pearl"]["context_sample_size_eval"]) != 256:
        raise ValueError("posterior adaptation requires context_sample_size_eval=256")
    if int(config["pearl"]["context_transitions_per_episode"]) != 32:
        raise ValueError("posterior adaptation requires context_transitions_per_episode=32")
    payload = {
        "schema": "posterior_adaptation_frozen_protocol_v1",
        "status": "preregistered_before_holdout",
        "config_hash": content_hash(config),
        "taskbook_hash": content_hash(taskbook_payload(taskbook)),
        "shots": shots,
        "support_selection": "fixed",
        "context_protocol": {
            "name": "fixed_nested_v1",
            "sample_size": 256,
            "transitions_per_episode": 32,
            "episode_capacity": 8,
        },
        "query_cases_per_task": int(config["evaluation"]["query_cases_per_task"]),
        "training_seeds": [int(value) for value in protocol["training_seeds"]],
        "required_methods": list(protocol["required_methods"]),
        "primary_metric": str(protocol["primary_metric"]),
        "primary_shot": int(protocol["primary_shot"]),
        "confidence_level": float(protocol["confidence_level"]),
        "bootstrap_samples": int(protocol["bootstrap_samples"]),
        "pass_criteria": dict(protocol["pass_criteria"]),
        "statistical_unit": "task",
        "inference": "task-cluster bootstrap with seed resampling within task",
        "uses_holdout_results": False,
        "source_config": str(Path(args.config)),
        "source_taskbook": str(Path(args.taskbook)),
    }
    write_json(args.output, payload)
    print(f"Posterior-adaptation protocol frozen before holdout: {args.output}")


if __name__ == "__main__":
    main()
