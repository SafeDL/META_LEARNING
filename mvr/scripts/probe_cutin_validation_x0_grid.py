"""Probe validation Cut-in query points without touching test or Outer paths."""
from __future__ import annotations

import json
import argparse
from itertools import product
from pathlib import Path

from ..experiments.cutin_inner import select_cutin_validation_tasks
from ..failure.criteria import FailureCriteria
from ..scenario.parameter_space import NormalizedScenarioAction
from ..scenario.taskbook import load_taskbook
from ..training.checkpoint import HierarchicalCheckpoint
from ..training.pipeline import build_model, checkpoint_config_hash, load_config
from .evaluate_cutin_inner_validation import _evaluate_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/cutin_inner.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated candidate labels to probe; empty means all labels",
    )
    args = parser.parse_args()
    config, taskbook_path, device = load_config(args.config)
    checkpoint = HierarchicalCheckpoint.load(
        args.checkpoint,
        expected_config_hash=checkpoint_config_hash(config),
    )
    model = build_model(config, device)
    model.load_state_dict(checkpoint.state["model"])
    model.eval()
    tasks = select_cutin_validation_tasks(
        load_taskbook(taskbook_path), config["cutin_inner"].get("validation_geometry_ids", ())
    )
    criteria = FailureCriteria.from_config(config["failure"])
    candidates = {
        "current": (-0.45, 0.45, 0.45, 0.45, 0.45, -0.45),
        "edge_a": (-0.25, 0.25, 0.25, 0.25, 0.25, -0.25),
        "edge_b": (-0.65, 0.65, 0.75, 0.65, 0.65, -0.65),
        "speed_gap": (-0.25, 0.65, 0.75, 0.65, 0.65, -0.25),
        "early": (-0.25, 0.25, 0.25, 0.25, 0.65, -0.25),
        "late": (-0.65, 0.65, 0.75, 0.65, 0.25, -0.65),
        "tight_early_fast": (-0.65, 0.65, 0.75, 0.25, 0.25, -0.65),
        "tight_late_fast": (-0.65, 0.65, 0.75, 0.65, 0.65, -0.65),
        "tight_early_slow": (-0.65, 0.25, 0.25, 0.25, 0.25, -0.65),
        "wide_early_fast": (-0.25, 0.65, 0.75, 0.25, 0.25, -0.65),
        "wide_late_fast": (-0.25, 0.65, 0.75, 0.65, 0.65, -0.65),
        "closing_mid": (-0.55, 0.35, 0.75, 0.45, 0.25, -0.55),
        "closing_late": (-0.55, 0.35, 0.75, 0.65, 0.45, -0.55),
        "closing_early": (-0.55, 0.35, 0.75, 0.25, 0.45, -0.55),
        "short_early": (-0.25, 0.35, 0.75, 0.25, 0.45, -0.65),
        "short_late": (-0.25, 0.35, 0.75, 0.65, 0.45, -0.65),
        "long_early": (-0.25, 0.35, 0.75, 0.25, 0.45, -0.25),
        "long_late": (-0.25, 0.35, 0.75, 0.65, 0.45, -0.25),
        "medium_close": (-0.45, 0.35, 0.75, 0.45, 0.45, -0.45),
        "medium_far": (-0.25, 0.35, 0.75, 0.45, 0.45, -0.45),
    }
    task = tasks[0]
    # Exhaustive corners of the validation Logical-domain box.  Values are
    # task-local normalized controls (not decoded physical parameters), so
    # every corner is a legal, reproducible 6D query.
    corner_values = tuple(
        tuple(
            float(lower + bit * (upper - lower))
            for (lower, upper), bit in zip(task.logical_domain_bounds.values(), bits)
        )
        for bits in product((0.0, 1.0), repeat=6)
    )
    candidates.update({
        f"corner_{index:02d}": value
        for index, value in enumerate(corner_values)
    })
    if args.labels:
        requested = [label.strip() for label in args.labels.split(",") if label.strip()]
        missing = sorted(set(requested).difference(candidates))
        if missing:
            raise ValueError(f"unknown candidate labels: {missing}")
        candidates = {label: candidates[label] for label in requested}
    rows: dict[str, object] = {}
    for label, values in candidates.items():
        query = NormalizedScenarioAction(args.candidate_index, values)
        records = _evaluate_task(
            model,
            task,
            criteria,
            shots=1,
            queries=1,
            max_support=4,
            seed=11,
            x0=query,
            step_budget=int(config["training"]["step_budget"]),
        )
        rows[label] = [
            {
                key: record[key]
                for key in (
                    "policy",
                    "event_kind",
                    "risk_score",
                    "continuous_risk_score",
                    "min_ttc",
                    "min_distance",
                    "max_closing_speed",
                )
            }
            for record in records
        ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
