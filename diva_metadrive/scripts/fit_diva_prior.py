"""Fit a v2 DIVA prior only after its frozen G0 structure gate passes."""
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
from ..provenance import content_hash
from .analyze_diva_prior import analyze_bank


def _load(path: str):
    return [
        observation_from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _support_indexes(bank: SourceBank, indexes: np.ndarray, shots: int, seed: int) -> np.ndarray:
    if shots == 0:
        return np.empty(0, dtype=int)
    per_candidate = {
        candidate:
        indexes[np.asarray([bank.designs[i].candidate_index == candidate for i in indexes])]
        for candidate in (0, 1)
    }
    if shots > len(indexes) or any(len(values) < shots // 2 for values in per_candidate.values()):
        raise ValueError("insufficient balanced common anchors for rank selection")
    rng = np.random.default_rng(seed)
    counts = (shots // 2, shots - shots // 2)
    selected = [
        rng.choice(per_candidate[candidate], size=counts[candidate], replace=False)
        for candidate in (0, 1)
    ]
    return np.sort(np.concatenate(selected))


def _select_rank(
    bank: SourceBank,
    candidates: tuple[int, ...],
    noise_var: float,
    *,
    support_shots: int = 4,
    seeds: tuple[int, ...] = (1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708),
) -> tuple[int, dict[str, dict[str, float]]]:
    """Deterministic balanced LOSO support selection, independent of design order."""
    losses: dict[int, list[float]] = {rank: [] for rank in candidates}
    squared_errors: dict[int, list[float]] = {rank: [] for rank in candidates}
    for held in range(len(bank.source_refs)):
        rows = [index for index in range(len(bank.source_refs)) if index != held]
        train = bank.responses[rows]
        train_eligible = bank.eligible[rows]
        refs = tuple(bank.source_refs[index] for index in rows)
        shared_common = train_eligible.all(axis=0) & bank.eligible[held]
        indexes = np.flatnonzero(shared_common)
        for rank in candidates:
            factor = fit_low_rank_vulnerability(train, train_eligible, refs, bank.design_ids, rank)
            for seed in seeds:
                support = _support_indexes(bank, indexes, support_shots, seed + 1009 * held)
                posterior = LatentVulnerabilityPosterior.standard_normal(rank)
                for index in support:
                    posterior.update(
                        factor.basis[index],
                        bank.responses[held, index] - factor.mean[index],
                        noise_var,
                    )
                for index in np.setdiff1d(indexes, support, assume_unique=True):
                    predicted, variance = posterior.predict(factor.mean[index],
                                                            factor.basis[index], noise_var)
                    residual = bank.responses[held, index] - predicted
                    losses[rank].append(
                        float(0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance)))
                    squared_errors[rank].append(float(residual**2))
    metrics = {
        str(rank): {
            "nll":
            float(np.mean(losses[rank])) if losses[rank] else float("inf"),
            "rmse":
            float(np.sqrt(np.mean(squared_errors[rank])))
            if squared_errors[rank] else float("inf"),
        }
        for rank in candidates
    }
    best = min(candidates, key=lambda rank: (metrics[str(rank)]["nll"], rank))
    values = np.asarray(losses[best], dtype=np.float64)
    standard_error = float(np.std(values, ddof=1) /
                           np.sqrt(len(values))) if len(values) > 1 else 0.0
    selected = min(rank for rank in candidates
                   if metrics[str(rank)]["nll"] <= metrics[str(best)]["nll"] + standard_error)
    return selected, metrics


def run(config_path: str, source_path: str, output_path: str, rank: int | None = None) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    observations = _load(source_path)
    bank = SourceBank.from_observations(observations, tuple(config["study"]["source_sut_refs"]))
    analysis = analyze_bank(bank, config, config["study"]["logical_domain_id"])
    if not analysis["g0_structural_viability"]["pass"]:
        raise ValueError("G0 failed; prior fitting and target evaluation are prohibited")
    bank.require_common_anchors(int(config["gates"]["g0"]["min_common_eligible_per_candidate"]))
    candidates = tuple(int(value) for value in config["prior"]["rank_candidates"])
    selected, selection = _select_rank(
        bank,
        candidates,
        float(config["prior"]["noise_floor_var"]),
        support_shots=int(config["prior"]["rank_selection_support_shots"]),
        seeds=tuple(int(seed) for seed in config["prior"]["rank_selection_seeds"]),
    )
    chosen = int(rank) if rank is not None else selected
    if chosen not in candidates:
        raise ValueError("requested rank is outside the frozen rank candidate set")
    prior = LowRankVulnerabilityPrior.fit(bank,
                                          chosen,
                                          device=str(config["device"]),
                                          gp_fit_steps=int(config["prior"]["gp_fit_steps"]))
    prior.noise_floor_var = float(config["prior"]["noise_floor_var"])
    prior.save(output_path)
    manifest = {
        "schema": "diva_prior_manifest_v2",
        "config_schema": config["schema"],
        "config_hash": content_hash(config),
        "rank": chosen,
        "rank_selection": {
            **{f"rank_{key}": value
               for key, value in selection.items()}, "selected_rank": selected
        },
        "source_refs": list(bank.source_refs),
        "source_bank_summary": bank.source_summary(),
        "singular_values": prior.factorization.singular_values.tolist(),
        "residual_variance": prior.factorization.residual_variance,
    }
    Path(output_path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) +
                                                               "\n",
                                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="diva_metadrive/configs/diva_cutin.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    run(args.config, args.source, args.output, args.rank)


if __name__ == "__main__":
    main()
