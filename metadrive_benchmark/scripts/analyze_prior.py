"""Analyze continuous source vulnerability structure without target access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..mining.source_bank import SourceBank, observation_from_dict


def _load(path: str):
    return [
        observation_from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def analyze_bank(bank: SourceBank, config: dict[str, Any], logical_domain_id: str) -> dict:
    common = bank.eligible.all(axis=0)
    if not common.any():
        raise ValueError("source bank has no common eligible anchors")
    values = bank.responses[:, common]
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = float(np.square(singular).sum())
    explained = (
        np.square(singular) / energy
        if energy > 1e-14
        else np.zeros_like(singular, dtype=np.float64)
    )
    per_source = {}
    for index, source in enumerate(bank.source_refs):
        response = values[index]
        per_source[source] = {
            "mean": float(np.mean(response)),
            "std": float(np.std(response)),
            "min": float(np.min(response)),
            "max": float(np.max(response)),
            "range": float(np.ptp(response)),
        }
    disagreement = np.std(values, axis=0)
    summary = bank.source_summary()
    common_by_candidate = {
        str(candidate): int(
            sum(
                common[index]
                for index, design in enumerate(bank.designs)
                if design.candidate_index == candidate
            )
        )
        for candidate in (0, 1)
    }
    gate = config["gates"]["g0"]
    criteria = {
        "formal_valid_rate": summary["formal_valid_rate"] >= gate["min_formal_valid_rate"],
        "common_eligible_per_candidate": all(
            count >= gate["min_common_eligible_per_candidate"]
            for count in common_by_candidate.values()
        ),
        "pooled_response_std": float(np.std(values)) >= gate["min_pooled_response_std"],
        "source_response_range": sum(
            stats["range"] >= gate["min_source_response_range"]
            for stats in per_source.values()
        ) >= gate["min_sources_meeting_response_range"],
        "anchor_disagreement_p75": float(np.percentile(disagreement, 75))
        >= gate["min_anchor_disagreement_p75"],
        "rank2_explained_variance": float(explained[:2].sum())
        >= gate["min_rank2_explained_variance"],
        "nondegenerate_centered_response": energy > 1e-14,
    }
    return {
        "schema": "mining_source_analysis_v2",
        "logical_domain_id": logical_domain_id,
        "summary": summary,
        "response_mean": float(np.mean(values)),
        "response_std": float(np.std(values)),
        "response_min": float(np.min(values)),
        "response_max": float(np.max(values)),
        "per_source_response_stats": per_source,
        "anchor_disagreement": _percentiles(disagreement),
        "singular_values": singular.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "rank1_cumulative_explained_variance": float(explained[:1].sum()),
        "rank2_cumulative_explained_variance": float(explained[:2].sum()),
        "common_anchor_count_by_candidate": common_by_candidate,
        "g0_structural_viability": {
            "pass": bool(all(criteria.values())),
            "criteria": criteria,
            "thresholds": gate,
            "failure_criteria": [name for name, passed in criteria.items() if not passed],
        },
    }


def _write_gate(output_path: str, report: dict) -> None:
    path = Path(output_path)
    gate_path = path.with_name(path.name.replace("source_analysis", "source_gate"))
    viability = report["g0_structural_viability"]
    gate = {
        "schema": "mining_source_gate_v2",
        "gate_id": "Mining-G0-source-structure",
        "logical_domain_id": report["logical_domain_id"],
        "decision": (
            "continue_to_prior_and_source_loso"
            if viability["pass"]
            else "stop_before_prior_fitting_and_target_evaluation"
        ),
        "g0_structural_viability": viability,
        "formal_event_statistics": {
            key: report["summary"][key]
            for key in (
                "formal_event_rate",
                "formal_collision_count",
                "formal_near_miss_count",
            )
        },
    }
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")


def run(config_path: str, source_path: str, output_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    observations = _load(source_path)
    domains = {observation.logical_domain_id for observation in observations}
    if domains != {config["study"]["logical_domain_id"]}:
        raise ValueError("source observations do not match the frozen v2 study domain")
    bank = SourceBank.from_observations(
        observations, tuple(config["study"]["source_sut_refs"])
    )
    report = analyze_bank(bank, config, domains.pop())
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_gate(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="metadrive_benchmark/configs/cutin.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.source, args.output)


if __name__ == "__main__":
    main()
