"""Build the E6 dual-mechanism Highway Cut-in response bank and Gate 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.generate_anchor_bank import generate_dual_mode_anchor_bank
from mvr.highway.data.response_bank import ResponseBank, build_response_bank
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.mining import critical_rewards, shared_prior_mining


def gate_zero(bank: ResponseBank, config: ExperimentConfig) -> dict:
    """Check difficulty, source structure, and target-specific score headroom."""
    config.validate()
    failure_rates = (bank.collisions | bank.near_misses).mean(axis=1)
    rank_two_evr = []
    shared_scores = []
    oracle_scores = []
    for target_index in range(len(bank.sut_names)):
        prior = LowRankPrior.fit(
            np.delete(bank.vulnerability, target_index, axis=0), config.prior_rank
        )
        rank_two_evr.append(float(prior.explained_variance_ratio[:2].sum()))
        shared_scores.append(
            shared_prior_mining(
                prior,
                bank.collisions[target_index],
                bank.near_misses[target_index],
                config.total_budget,
            ).critical_score
        )
        rewards = critical_rewards(
            bank.collisions[target_index], bank.near_misses[target_index]
        )
        oracle_scores.append(float(np.sort(rewards)[-config.total_budget :].sum()))
    headroom_passes = [
        oracle >= 1.10 * shared
        for oracle, shared in zip(oracle_scores, shared_scores, strict=True)
    ]
    return {
        "schema": "highway_diva_mine_e6_gate_zero_v1",
        "bank_shape": list(bank.vulnerability.shape),
        "mode_counts": {
            mode: int(np.sum(bank.modes == mode))
            for mode in sorted(set(np.asarray(bank.modes).tolist()))
        },
        "per_sut": [
            {
                "target_sut": name,
                "failure_rate": float(failure_rate),
                "rank_two_explained_variance_ratio": evr,
                "shared_critical_score_at_20": shared,
                "oracle_critical_score_at_20": oracle,
                "oracle_score_relative_gain": (oracle - shared) / shared if shared else None,
                "oracle_score_headroom_at_least_10_percent": passed,
            }
            for name, failure_rate, evr, shared, oracle, passed in zip(
                bank.sut_names,
                failure_rates,
                rank_two_evr,
                shared_scores,
                oracle_scores,
                headroom_passes,
                strict=True,
            )
        ],
        "acceptance": {
            "failure_rate_5_to_50_percent_for_every_sut": bool(
                np.all((failure_rates >= 0.05) & (failure_rates <= 0.50))
            ),
            "rank_2_evr_at_least_80_percent_for_every_loso_fold": bool(
                np.all(np.asarray(rank_two_evr) >= 0.80)
            ),
            "oracle_score_headroom_at_least_10_percent_for_4_of_6_suts": sum(headroom_passes)
            >= 4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/diva_highway/cutin_mvp_e6")
    )
    parser.add_argument("--rebuild-bank", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig()
    bank_path = args.output_dir / "response_bank_highway_e6.npz"
    if args.rebuild_bank or not bank_path.exists():
        anchors, modes = generate_dual_mode_anchor_bank(config.num_anchors, config.seed)
        bank = build_response_bank(anchors, config.seed, modes)
        bank.save(bank_path)
    else:
        bank = ResponseBank.load(bank_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate = gate_zero(bank, config)
    gate_path = args.output_dir / "gate_zero_e6.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
