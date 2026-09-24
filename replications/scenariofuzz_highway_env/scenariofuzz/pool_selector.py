"""Separate fixed-pool SEM selector; this is not online ScenarioFuzz-H."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sut_algorithms.highway_env.idm_profiles import get_profile

from .corpus import ScenarioSpec, build_default_corpus
from .execution import execute_scenario
from .filter import load_checkpoint, predict_scores
from .io_utils import file_hash, load_config, write_csv
from .sem_training import load_source_history


@dataclass
class SEMPoolSelector:
    scores: np.ndarray
    eligible: np.ndarray
    selected: set[int]

    @classmethod
    def initialize(cls, scores: np.ndarray, eligible_ids: np.ndarray) -> "SEMPoolSelector":
        return cls(np.asarray(scores, dtype=float), np.asarray(eligible_ids, dtype=int), set())

    def propose(self) -> int | None:
        available = [i for i in self.eligible if int(i) not in self.selected]
        if not available:
            return None
        choice = int(max(available, key=lambda i: self.scores[int(i)]))
        self.selected.add(choice)
        return choice


def run_pool(config_path: Path, source_bank: Path, sem_path: Path, output: Path) -> None:
    """Rank a frozen source-defined pool and reveal each target outcome once."""
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    source = load_source_history(source_bank)
    unique: dict[str, ScenarioSpec] = {}
    for scenario_id, gap, speed, mode in zip(source["scenario_id"], source["initial_gap"], source["relative_speed"], source["mode"]):
        unique.setdefault(str(scenario_id), ScenarioSpec(str(scenario_id), float(gap), float(speed), str(mode)))
    specs = list(unique.values())
    seed = build_default_corpus(config)[0]
    model, _ = load_checkpoint(sem_path, "cuda" if __import__("torch").cuda.is_available() else "cpu")
    scores = predict_scores(model, seed, specs, next(model.parameters()).device.type)
    selector = SEMPoolSelector.initialize(scores, np.arange(len(specs)))
    profile = get_profile(config["target_sut"])
    rows = []
    for step in range(1, int(config["budget"]) + 1):
        index = selector.propose()
        if index is None:
            break
        spec = specs[index]
        observation = execute_scenario(profile, spec, int(config["random_seed"]) + step, output / "trajectories")
        rows.append({
            "step": step, "scenario_id": spec.scenario_id, "initial_gap": spec.initial_gap,
            "relative_speed": spec.relative_speed, "mode": spec.mode,
            "method_variant": "SEM-Pool-H", "task_kind": "failure_discovery",
            "predicted_score": float(scores[index]), "score_semantics": "source-only SEM historical failure tendency",
            "visible_target_count": step - 1, "history_split_id": "source_only_grouped_v1",
            "selected_reason": "highest_unqueried_fixed_pool_score", "observed_event": observation.collision,
            "valid": observation.valid, "collision": observation.collision, "near_miss": observation.near_miss,
            "cumulative_cost": step, "execution_id": observation.execution_id, "trajectory_path": observation.trajectory_path,
        })
    write_csv(output / "selection_log.csv", rows)
    summary = {
        "protocol": "SEM-Pool-H", "candidate_pool_size": len(specs), "target_sut": profile.name,
        "target_executions": len(rows), "collisions": int(sum(row["collision"] for row in rows)),
        "critical_events": int(sum(row["collision"] or row["near_miss"] for row in rows)),
        "first_collision_budget": next((row["step"] for row in rows if row["collision"]), None),
        "source_bank_sha256": file_hash(source_bank), "sem_checkpoint_sha256": file_hash(sem_path),
        "new_scenarios_generated": 0, "online_fuzzing_claimed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--sem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run_pool(args.config, args.source_bank, args.sem, args.output)
    print(f"Wrote separate fixed-pool protocol to {args.output}")


if __name__ == "__main__":
    main()
