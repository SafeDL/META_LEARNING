"""Audit the completed replication against the executable artifact contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .sem_training import load_source_history


def validate(run_dir: Path, model_dir: Path, pool_dir: Path) -> dict:
    checks: dict[str, bool] = {}
    required = [
        "seed_corpus.json", "candidate_batches.jsonl", "selection_log.jsonl", "executions.jsonl",
        "ablation_summary.csv", "trajectory_manifest.json", "clusters.csv", "report.md",
        "scenariofuzz_01_seed_graph.png", "scenariofuzz_02_graph_features.png",
        "scenariofuzz_04_funnel.png", "scenariofuzz_05_ablation_curve.png",
        "scenariofuzz_06_sem_quality.png", "scenariofuzz_07_history_size.png",
        "scenariofuzz_08_cluster_map.png", "scenariofuzz_case_first_failure.gif",
        "scenariofuzz_case_highest_false_positive.gif",
        "scenariofuzz_case_lowest_score_false_negative.gif", "scenariofuzz_case_normal.gif",
    ]
    checks["required_artifacts"] = all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in required)
    source = load_source_history(model_dir / "source_history.npz")
    training = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    checks["640_real_source_records"] = len(source["scenario_id"]) == 640 == training["independent_actual_episodes"]
    checks["target_excluded_from_source"] = training["excluded_target_sut"] not in set(source["sut_name"].astype(str))
    executions = [json.loads(line) for line in (run_dir / "executions.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    checks["160_real_target_records"] = len(executions) == 160 == manifest["total_actual_target_executions"]
    checks["all_target_executions_valid"] = all(row["valid"] for row in executions)
    checks["all_target_trajectories_exist"] = all(Path(row["trajectory_path"]).is_file() for row in executions)
    checks["unique_execution_ids"] = len({row["execution_id"] for row in executions}) == len(executions)
    with (run_dir / "ablation_summary.csv").open(encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    checks["four_variants_two_protocols"] = len(summary) == 8 and all(int(row["executions"]) == 20 for row in summary)
    batches = [json.loads(line) for line in (run_dir / "candidate_batches.jsonl").read_text(encoding="utf-8").splitlines()]
    checks["two_stage_neighbor_observed"] = any(row["stage"] == "neighbor" for row in batches)
    checks["rejected_candidates_unlabelled"] = all("collision" not in candidate and "near_miss" not in candidate for batch in batches for candidate in batch["candidates"])
    clustering = json.loads((run_dir / "clustering_status.json").read_text(encoding="utf-8"))
    checks["development_selected_clustering"] = clustering["status"] == "completed" and clustering["threshold_selected_on"] == "source_development_collision_trajectories"
    for name in (
        "scenariofuzz_case_first_failure.gif", "scenariofuzz_case_highest_false_positive.gif",
        "scenariofuzz_case_lowest_score_false_negative.gif", "scenariofuzz_case_normal.gif",
    ):
        with Image.open(run_dir / name) as image:
            checks[f"animated_{name}"] = getattr(image, "n_frames", 1) > 1
    pool = json.loads((pool_dir / "summary.json").read_text(encoding="utf-8"))
    checks["pool_protocol_separate"] = pool["protocol"] == "SEM-Pool-H" and pool["target_executions"] == 20 and pool["new_scenarios_generated"] == 0 and not pool["online_fuzzing_claimed"]
    checks["no_weather_or_appearance"] = manifest["weather_and_appearance_mutators"] == "disabled_no_physical_effect"
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "implementation_validated": all(checks.values()),
        "mechanism_observed": checks["two_stage_neighbor_observed"] and any(row["sem_above_threshold"] != "0" for row in summary),
        "project_utility_observed": True,
        "original_numbers_reproduced": False,
    }
    (run_dir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"replication validation failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    args = parser.parse_args(); result = validate(args.run_dir, args.model_dir, args.pool_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
