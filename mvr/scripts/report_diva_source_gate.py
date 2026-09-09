"""Summarize the preregistered DIVA source-only domain-selection gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _elapsed_seconds(observation_path: str) -> list[float]:
    rows = Path(observation_path).read_text(encoding="utf-8").splitlines()
    return [float(json.loads(row)["elapsed_seconds"]) for row in rows if row]


def run(analysis_paths: list[str], observation_paths: list[str], output_path: str) -> dict:
    if len(analysis_paths) != len(observation_paths):
        raise ValueError("each analysis file requires its matching observation file")

    domains = []
    durations: list[float] = []
    for analysis_path, observation_path in zip(analysis_paths, observation_paths):
        report = _load_json(analysis_path)
        summary = report["summary"]
        headroom = report["headroom"]
        durations.extend(_elapsed_seconds(observation_path))
        domains.append(
            {
                "logical_domain_id": report.get(
                    "logical_domain_id",
                    Path(analysis_path).stem.removesuffix("_analysis"),
                ),
                "physical_calls": summary["observations"],
                "g1_source_viability": report["g1_source_viability"],
                "formal_valid_rate": summary["formal_valid_rate"],
                "posterior_eligible_rate": summary["posterior_eligible_rate"],
                "event_rate": summary["event_rate"],
                "common_eligible_anchors": summary["common_eligible_anchors"],
                "minimum_oracle_top8_gain": min(item["gain"] for item in headroom),
                "mean_oracle_top8_gain": float(np.mean([item["gain"] for item in headroom])),
                "analysis_path": analysis_path,
                "analysis_sha256": _sha256(analysis_path),
                "observations_path": observation_path,
                "observations_sha256": _sha256(observation_path),
            }
        )

    passing = [domain for domain in domains if domain["g1_source_viability"]["pass"]]
    report = {
        "schema": "diva_source_domain_gate",
        "gate_id": "DIVA-G1-source-domain-selection",
        "physical_calls": int(sum(domain["physical_calls"] for domain in domains)),
        "duration_seconds": {
            "median": float(np.median(durations)),
            "p90": float(np.percentile(durations, 90)),
        },
        "domains": domains,
        "selected_domain": None,
        "decision": (
            "stop_before_source_expansion_and_target_evaluation"
            if not passing
            else "continue_with_selected_domain"
        ),
        "reason": (
            "No preregistered domain satisfies G1; no source response matrix contains both "
            "eligible zero and positive outcomes for every source and candidate."
            if not passing
            else "At least one preregistered domain satisfies G1."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", nargs="+", required=True)
    parser.add_argument("--observations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.analysis, args.observations, args.output)


if __name__ == "__main__":
    main()
