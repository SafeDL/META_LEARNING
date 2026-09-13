"""Build the E6 dual-mechanism Highway Cut-in response bank and Gate 0 audit."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.generate_anchor_bank import generate_dual_mode_anchor_bank
from mvr.highway.data.response_bank import ResponseBank, build_response_bank
from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.mining import critical_rewards, shared_prior_mining


GATE_SCHEMA = "highway_diva_mine_e6_gate_zero_v2_action_fix"
OUTPUT_DIR = Path("results/diva_highway/cutin_mvp_e6_action_fix")
VALIDATION_PATH = OUTPUT_DIR / "mechanism_validation.json"
MIN_HEADROOM_TARGETS = 4


def _relative_gain(oracle_score: float, shared_score: float) -> float | None:
    return (oracle_score - shared_score) / shared_score if shared_score else None


def gate_zero(bank: ResponseBank, config: ExperimentConfig) -> dict:
    """Check difficulty, source structure, and target-specific score headroom."""
    config.validate()
    if bank.modes is None:
        raise ValueError("Gate 0 requires an interaction mode for every anchor")
    modes = np.asarray(bank.modes)
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
        "schema": GATE_SCHEMA,
        "bank_shape": list(bank.vulnerability.shape),
        "mode_counts": {
            mode: int(np.sum(modes == mode)) for mode in sorted(set(modes.tolist()))
        },
        "per_sut": [
            {
                "target_sut": name,
                "failure_rate": float(failure_rate),
                "rank_two_explained_variance_ratio": evr,
                "shared_critical_score_at_20": shared,
                "oracle_critical_score_at_20": oracle,
                "oracle_score_relative_gain": _relative_gain(oracle, shared),
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
            "oracle_score_headroom_at_least_10_percent_for_4_of_6_suts": (
                sum(headroom_passes) >= MIN_HEADROOM_TARGETS
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
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
    gate["generation"] = {
        "highway_env_version": version("highway-env"),
        "scheduled_lead_action": "Vehicle.act direct low-level action",
        "mode_contract": {
            "fast_intrusion": "0.45 s scheduled lane change, constant target speed",
            "cutin_braking": "1.5 s scheduled lane change, then -4.5 m/s^2 for 1.0 s",
        },
        "mechanism_validation": str(VALIDATION_PATH).replace("\\", "/"),
    }
    gate_path = args.output_dir / "gate_zero_e6.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
