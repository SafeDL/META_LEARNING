"""Create a source-only Sobol casebook for retained-domain DIVA Cut-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..provenance import content_hash
from ..scenario.taskbook import load_taskbook
from ..diva.source_bank import retained_task, sobol_designs


def run(
    config_path: str,
    output_path: str,
    anchors_per_candidate: int | None = None,
    logical_domain_id: str | None = None,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    study = config["study"]
    tasks = load_taskbook(config["taskbook"])
    source_refs = tuple(study["source_sut_refs"])
    domain = str(logical_domain_id or study["logical_domain_id"])
    selected = [retained_task(tasks, source, domain) for source in source_refs]
    baseline = selected[0]
    if any(task.geometry_hash != baseline.geometry_hash for task in selected):
        raise ValueError("DIVA source tasks must retain one physical geometry")
    count = int(anchors_per_candidate or config["source_bank"]["anchors_per_candidate"])
    designs = sobol_designs(baseline, count, int(config["seed"]))
    payload = {
        "schema": "diva_cutin_casebook_constant_speed_physical",
        "config_hash": content_hash(config),
        "geometry_id": baseline.geometry_id,
        "geometry_hash": baseline.geometry_hash,
        "logical_domain_id": domain,
        "source_sut_refs": list(source_refs),
        "anchors_per_candidate": count,
        "designs": [design.to_dict() for design in designs],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_cutin.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors-per-candidate", type=int)
    parser.add_argument("--logical-domain-id")
    args = parser.parse_args()
    run(args.config, args.output, args.anchors_per_candidate, args.logical_domain_id)


if __name__ == "__main__":
    main()
