"""Fit and persist a frozen DIVA prior from source-only observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..diva.factorization import fit_low_rank_vulnerability
from ..diva.posterior import LatentVulnerabilityPosterior
from ..diva.prior import LowRankVulnerabilityPrior
from ..diva.source_bank import SourceBank, observation_from_dict


def _load(path: str):
    return [observation_from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _select_rank(bank: SourceBank, candidates: tuple[int, ...], noise_var: float) -> tuple[int, dict[str, float]]:
    """LOSO source-only working-model NLL with a fixed nested four-anchor prefix."""
    losses: dict[int, list[float]] = {rank: [] for rank in candidates}
    for held in range(len(bank.source_refs)):
        rows = [index for index in range(len(bank.source_refs)) if index != held]
        train_scores, train_eligible = bank.scores[rows], bank.eligible[rows]
        train_refs = tuple(bank.source_refs[index] for index in rows)
        for rank in candidates:
            factor = fit_low_rank_vulnerability(train_scores, train_eligible, train_refs, bank.design_ids, rank)
            common = factor.common_eligible_mask & bank.eligible[held]
            eligible_indexes = np.flatnonzero(common)
            if len(eligible_indexes) <= 4:
                continue
            posterior = LatentVulnerabilityPosterior.standard_normal(rank)
            for index in eligible_indexes[:4]:
                posterior.update(factor.basis[index], bank.scores[held, index] - factor.mean[index], noise_var)
            query = eligible_indexes[4:]
            for index in query:
                predicted, variance = posterior.predict(
                    factor.mean[index], factor.basis[index], noise_var
                )
                residual = bank.scores[held, index] - predicted
                losses[rank].append(
                    float(0.5 * (np.log(2.0 * np.pi * variance) + residual ** 2 / variance))
                )
    means = {str(rank): float(np.mean(values)) if values else float("inf") for rank, values in losses.items()}
    ordered = sorted(candidates, key=lambda rank: (means[str(rank)], rank))
    best = ordered[0]
    # One-standard-error simplification: choose the smaller rank when its LOSO
    # MSE is within one empirical standard error of the best candidate.
    best_values = losses[best]
    threshold = means[str(best)] + (float(np.std(best_values, ddof=1)) / np.sqrt(len(best_values)) if len(best_values) > 1 else 0.0)
    selected = min(rank for rank in candidates if means[str(rank)] <= threshold)
    return selected, means


def run(config_path: str, source_path: str, output_path: str, rank: int | None = None) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    bank = SourceBank.from_observations(_load(source_path), tuple(config["study"]["source_sut_refs"]))
    bank.require_common_anchors()
    bank.require_response_boundary()
    candidates = tuple(int(value) for value in config["prior"]["rank_candidates"])
    selected, losses = _select_rank(bank, candidates, float(config["prior"]["noise_floor_var"]))
    chosen = int(rank) if rank is not None else selected
    if chosen not in candidates:
        raise ValueError("requested rank is outside the frozen rank candidate set")
    prior = LowRankVulnerabilityPrior.fit(
        bank,
        chosen,
        device=str(config["device"]),
        gp_fit_steps=int(config["prior"]["gp_fit_steps"]),
    )
    prior.noise_floor_var = float(config["prior"]["noise_floor_var"])
    prior.save(output_path)
    manifest = {
        "schema": "diva_prior_manifest",
        "rank": chosen,
        "rank_loso_nll": losses,
        "source_refs": list(bank.source_refs),
        "source_bank_summary": bank.source_summary(),
        "singular_values": prior.factorization.singular_values.tolist(),
        "residual_variance": prior.factorization.residual_variance,
    }
    manifest_path = Path(output_path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_cutin.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    run(args.config, args.source, args.output, args.rank)


if __name__ == "__main__":
    main()
