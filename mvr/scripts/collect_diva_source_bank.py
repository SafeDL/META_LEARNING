"""Execute a frozen DIVA source casebook and append every physical observation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..diva.episode_executor import DivaEpisodeExecutor
from ..diva.source_bank import retained_task
from ..diva.types import DivaCutInDesign
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner


def _casebook(casebook_path: str) -> tuple[str, tuple[DivaCutInDesign, ...]]:
    payload = json.loads(Path(casebook_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "diva_cutin_casebook_constant_speed_physical":
        raise ValueError("unsupported DIVA casebook schema")
    designs = []
    for row in payload["designs"]:
        row = dict(row)
        row.pop("schema", None)
        row.pop("design_id", None)
        designs.append(DivaCutInDesign(**row))
    return str(payload["logical_domain_id"]), tuple(designs)


def run(config_path: str, casebook_path: str, output_path: str) -> int:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tasks = load_taskbook(config["taskbook"])
    study = config["study"]
    source_refs = tuple(study["source_sut_refs"])
    domain, designs = _casebook(casebook_path)
    selected = [retained_task(tasks, source, domain) for source in source_refs]
    seeds = tuple(int(value) for value in config["source_bank"]["seeds"])
    executor = DivaEpisodeExecutor(
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()),
        HierarchicalRunner(int(config["execution"]["runner_step_budget"])),
        int(config["execution"]["environment_horizon"]),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as stream:
        for task in selected:
            for design_index, design in enumerate(designs):
                for seed in seeds:
                    episode_seed = int(seed + 10_000 * design_index)
                    observation = executor.run(task, design, episode_seed)
                    stream.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")
                    stream.flush()
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_cutin.yaml")
    parser.add_argument("--casebook", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.casebook, args.output)


if __name__ == "__main__":
    main()
