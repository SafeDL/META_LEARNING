"""Create the frozen v2 Mining source casebook from study physical bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..mining.source_bank import retained_task, sobol_designs_from_physical_bounds
from ..mining.types import MINING_SCHEMA
from ..provenance import content_hash
from ..scenario.catalog import mvr_parameter_spaces, valid_cutin_initial_state
from ..scenario.task_spec import logical_parameter_names
from ..scenario.taskbook import load_taskbook


CASEBOOK_SCHEMA = "mining_cutin_source_casebook_v2"


def _validate_designs(designs, bounds) -> None:
    if len(designs) != 64 or sum(d.candidate_index == 0 for d in designs) != 32:
        raise ValueError("v2 source casebook must contain 32 designs per candidate")
    space = mvr_parameter_spaces()["cutin"]
    names = logical_parameter_names("cutin")
    if len({design.design_id for design in designs}) != len(designs):
        raise ValueError("v2 source casebook designs must be unique")
    for design in designs:
        values = space.decode(design.scenario_action())
        for name in names:
            lower, upper = bounds[name]
            if not lower <= float(values[name]) <= upper:
                raise ValueError("Mining design escaped frozen source physical bounds")
        if not valid_cutin_initial_state(
            float(values["ego_initial_speed_mps"]),
            float(values["relative_speed_mps"]),
            float(values["initial_gap_m"]),
            float(values["cutin_path_length_m"]),
        ):
            raise ValueError("Mining design violates Cut-in physical reset constraints")


def run(config_path: str, output_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    study = config["study"]
    tasks = load_taskbook(config["taskbook"])
    task_domain = study["task_logical_domain_id"]
    selected = [retained_task(tasks, source, task_domain) for source in study["source_sut_refs"]]
    baseline = selected[0]
    if any(task.geometry_hash != baseline.geometry_hash for task in selected):
        raise ValueError("Mining source tasks must retain one physical geometry")
    bounds = study["source_physical_bounds"]
    designs = sobol_designs_from_physical_bounds(
        bounds, int(config["source_bank"]["anchors_per_candidate"]), int(config["seed"])
    )
    _validate_designs(designs, bounds)
    payload = {
        "schema": CASEBOOK_SCHEMA,
        "observation_schema": MINING_SCHEMA,
        "config_hash": content_hash(config),
        "geometry_id": baseline.geometry_id,
        "geometry_hash": baseline.geometry_hash,
        "logical_domain_id": study["logical_domain_id"],
        "task_logical_domain_id": task_domain,
        "source_physical_bounds": bounds,
        "source_sut_refs": list(study["source_sut_refs"]),
        "anchors_per_candidate": int(config["source_bank"]["anchors_per_candidate"]),
        "designs": [design.to_dict() for design in designs],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="metadrive_sim_env/configs/cutin.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
