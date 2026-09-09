"""Analyze source-only DIVA viability and low-rank headroom without target access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..diva.factorization import fit_low_rank_vulnerability
from ..diva.source_bank import SourceBank, observation_from_dict


def _load(path: str):
    return [observation_from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def run(config_path: str, source_path: str, output_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    observations = _load(source_path)
    domains = {observation.logical_domain_id for observation in observations}
    if len(domains) != 1:
        raise ValueError("source analysis requires exactly one logical domain")
    bank = SourceBank.from_observations(observations, tuple(config["study"]["source_sut_refs"]))
    common = bank.eligible.all(axis=0)
    if not common.any():
        raise ValueError("source bank has no common eligible anchors")
    headroom = []
    for source_index, source in enumerate(bank.source_refs):
        values = bank.scores[source_index, common]
        others = [index for index in range(len(bank.source_refs)) if index != source_index]
        shared = np.nanmean(bank.scores[others][:, common], axis=0)
        oracle = np.sort(values)[-min(8, len(values)):].sum()
        shared_selected = values[np.argsort(shared)[-min(8, len(values)):]].sum()
        headroom.append({"sut_ref": source, "oracle_top8": float(oracle), "shared_top8": float(shared_selected), "gain": float(oracle - shared_selected)})
    full = fit_low_rank_vulnerability(bank.scores, bank.eligible, bank.source_refs, bank.design_ids, min(3, len(bank.source_refs) - 1))
    summary = bank.source_summary()
    boundary = summary["response_boundary_counts"]
    viable_boundary = all(
        values["eligible_zero"] > 0 and values["eligible_positive"] > 0
        for candidates in boundary.values() for values in candidates.values()
    )
    viable = bool(
        summary["formal_valid_rate"] >= 0.80
        and 0.02 <= summary["event_rate"] <= 0.80
        and viable_boundary
    )
    report = {
        "schema": "diva_source_analysis",
        "logical_domain_id": domains.pop(),
        "summary": summary,
        "g1_source_viability": {
            "pass": viable,
            "formal_valid_rate_minimum": 0.80,
            "event_rate_interval": [0.02, 0.80],
            "requires_eligible_zero_and_positive_per_source_candidate": True,
            "failure_reason": None if viable else "source response boundary is not evaluable",
        },
        "singular_values": full.singular_values.tolist(),
        "headroom": headroom,
        "common_anchor_count_by_candidate": {
            str(candidate): int(sum(common[index] for index, design in enumerate(bank.designs) if design.candidate_index == candidate))
            for candidate in (0, 1)
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_cutin.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.source, args.output)


if __name__ == "__main__":
    main()
