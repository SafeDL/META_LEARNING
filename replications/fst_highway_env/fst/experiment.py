"""Freeze FST sets, then evaluate held-out highway-env SUTs and baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from diva_highway_env.data.response_bank import ResponseBank

from .fluctuation_audit import signed_fluctuation
from .fusion import inverse_distance_attention, weights_from_attention
from .reference_distribution import scenario_ids, sha256_file, uniform_distribution
from .response_clusters import ResponseClusterSampler, sample_uniform
from .set_optimizer import discrete_single_swap
from .similarity_network import (
    FeatureTransform,
    SimilarityNetwork,
    minimax_loss_per_set,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(run_dir: Path) -> tuple[SimilarityNetwork, FeatureTransform]:
    checkpoint = torch.load(run_dir / "similarity_model.pt", map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    config.pop("similarity", None)
    model = SimilarityNetwork(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, FeatureTransform.from_dict(checkpoint["feature_transform"])


def _learned_components(
    model: SimilarityNetwork,
    feature_tensor: torch.Tensor,
    probability_tensor: torch.Tensor,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        embedding = model.encode(feature_tensor)
        attention = model.attention_from_embeddings(
            embedding, torch.as_tensor(indices, dtype=torch.long)
        )[0]
        weights = attention @ probability_tensor
    return attention.numpy(), weights.numpy()


def _learned_loss_function(
    model: SimilarityNetwork,
    feature_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    probability_tensor: torch.Tensor,
):
    with torch.no_grad():
        embedding = model.encode(feature_tensor)

    def evaluate(sets: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for chunk in np.array_split(np.asarray(sets, dtype=int), max(1, int(np.ceil(len(sets) / 256)))):
            if not len(chunk):
                continue
            index_tensor = torch.as_tensor(chunk, dtype=torch.long)
            with torch.no_grad():
                attention = model.attention_from_embeddings(embedding, index_tensor)
                weights = torch.einsum("bnl,l->bn", attention, probability_tensor)
                loss, _ = minimax_loss_per_set(
                    weights, source_tensor, index_tensor, probability_tensor
                )
            values.append(loss.numpy())
        return np.concatenate(values)

    return evaluate


def _handcrafted_components(
    features: np.ndarray, probability: np.ndarray, indices: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    attention = inverse_distance_attention(features, indices, epsilon=epsilon)
    return attention, weights_from_attention(attention, probability)


def _handcrafted_loss_function(
    features: np.ndarray, source_response: np.ndarray, probability: np.ndarray, epsilon: float
):
    truth = source_response @ probability

    def evaluate(sets: np.ndarray) -> np.ndarray:
        selected = features[np.asarray(sets, dtype=int)]
        distances = np.linalg.norm(selected[:, :, None, :] - features[None, None, :, :], axis=3)
        logits = 1.0 / (distances + epsilon)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        attention = exp / exp.sum(axis=1, keepdims=True)
        weights = np.einsum("bnl,l->bn", attention, probability)
        responses = source_response[:, np.asarray(sets, dtype=int)].transpose(1, 0, 2)
        estimates = np.einsum("bn,bmn->bm", weights, responses)
        return np.max(np.abs(estimates - truth[None, :]), axis=1)

    return evaluate


def _plain_loss_function(source_response: np.ndarray, probability: np.ndarray):
    truth = source_response @ probability

    def evaluate(sets: np.ndarray) -> np.ndarray:
        values = source_response[:, np.asarray(sets, dtype=int)].mean(axis=2).T
        return np.max(np.abs(values - truth[None, :]), axis=1)

    return evaluate


def _design_record(
    method: str,
    budget: int,
    repeat: int,
    indices: np.ndarray,
    weights: np.ndarray,
    source_response: np.ndarray,
    probability: np.ndarray,
    scenario_identifiers: list[str],
    source_loss: float | None = None,
) -> dict[str, object]:
    truth = source_response @ probability
    estimate = source_response[:, indices] @ weights
    return {
        "method": method,
        "budget": budget,
        "repeat": repeat,
        "indices": [int(v) for v in indices],
        "scenario_ids": [scenario_identifiers[int(v)] for v in indices],
        "weights": [float(v) for v in weights],
        "source_minimax_loss": float(np.max(np.abs(estimate - truth)) if source_loss is None else source_loss),
    }


def run_experiment(run_dir: Path, budgets: list[int] | None = None) -> dict[str, object]:
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    train_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bank_path = Path(train_manifest["bank_path"])
    if sha256_file(bank_path) != train_manifest["bank_sha256"]:
        raise RuntimeError("response bank changed after training")
    bank = ResponseBank.load(bank_path)
    model, transform = _load_model(run_dir)
    modes = np.asarray(bank.modes).astype(str)
    identifiers = scenario_ids(bank.anchors, modes)
    probability = uniform_distribution(len(bank.anchors))
    features = transform.transform(bank.anchors, modes)
    feature_tensor = torch.as_tensor(features)
    probability_tensor = torch.as_tensor(probability, dtype=torch.float32)
    source_names = [str(v) for v in config["source_train_suts"] + config["source_dev_suts"]]
    source_rows = np.asarray([bank.index_of(name) for name in source_names], dtype=int)
    source_response = bank.collisions[source_rows].astype(float)
    source_tensor = torch.as_tensor(source_response, dtype=torch.float32)
    test_budgets = [int(v) for v in (budgets or config["test_n"])]
    repetitions = int(config["experiment_repetitions"])
    seed = int(config["seed"])
    cluster_sampler = ResponseClusterSampler.fit(
        source_response, int(config["response_clusters"]), seed
    )
    learned_loss = _learned_loss_function(
        model, feature_tensor, source_tensor, probability_tensor
    )
    handcrafted_loss = _handcrafted_loss_function(
        features[:, :2], source_response, probability, float(config["distance_epsilon"])
    )
    plain_loss = _plain_loss_function(source_response, probability)
    source_risk = source_response.mean(axis=0)
    if float(source_risk.sum()) == 0.0:
        proposal = probability.copy()
    else:
        proposal = 0.9 * source_risk / source_risk.sum() + 0.1 * probability

    designs: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    primary_attention: dict[str, np.ndarray] = {}
    primary_weights: dict[str, np.ndarray] = {}
    primary_indices: dict[str, np.ndarray] = {}
    design_started = time.perf_counter()
    total_candidate_evaluations = 0
    for budget in test_budgets:
        for repeat in range(repetitions):
            rng = np.random.default_rng(seed + budget * 10000 + repeat)
            initial = cluster_sampler.sample(budget, rng)

            cmc_indices = rng.choice(len(probability), size=budget, replace=True, p=probability)
            designs.append(_design_record(
                "CMC", budget, repeat, cmc_indices, np.full(budget, 1.0 / budget),
                source_response, probability, identifiers,
            ))
            uniform_indices = sample_uniform(budget, len(probability), rng)
            designs.append(_design_record(
                "Uniform", budget, repeat, uniform_indices, np.full(budget, 1.0 / budget),
                source_response, probability, identifiers,
            ))
            is_indices = rng.choice(len(probability), size=budget, replace=True, p=proposal)
            is_weights = probability[is_indices] / proposal[is_indices] / budget
            designs.append(_design_record(
                "IS", budget, repeat, is_indices, is_weights,
                source_response, probability, identifiers,
            ))

            _random_attention, random_weights = _learned_components(
                model, feature_tensor, probability_tensor, initial
            )
            designs.append(_design_record(
                "FST-RandomSet", budget, repeat, initial, random_weights,
                source_response, probability, identifiers,
            ))

            handcrafted_result = discrete_single_swap(
                initial, len(probability), handcrafted_loss,
                max_rounds=int(config["max_swap_rounds"]),
            )
            _handcrafted_attention, handcrafted_weights = _handcrafted_components(
                features[:, :2], probability, handcrafted_result.indices,
                float(config["distance_epsilon"]),
            )
            designs.append(_design_record(
                "Handcrafted-Similarity", budget, repeat, handcrafted_result.indices,
                handcrafted_weights, source_response, probability, identifiers,
                handcrafted_result.loss,
            ))
            total_candidate_evaluations += handcrafted_result.candidate_evaluations

            learned_result = discrete_single_swap(
                initial, len(probability), learned_loss,
                max_rounds=int(config["max_swap_rounds"]),
            )
            learned_attention, learned_weights = _learned_components(
                model, feature_tensor, probability_tensor, learned_result.indices
            )
            designs.append(_design_record(
                "FST-Similarity-H", budget, repeat, learned_result.indices,
                learned_weights, source_response, probability, identifiers,
                learned_result.loss,
            ))
            total_candidate_evaluations += learned_result.candidate_evaluations

            plain_result = discrete_single_swap(
                initial, len(probability), plain_loss,
                max_rounds=int(config["max_swap_rounds"]),
            )
            designs.append(_design_record(
                "NoSimilarity-Optimized", budget, repeat, plain_result.indices,
                np.full(budget, 1.0 / budget), source_response, probability, identifiers,
                plain_result.loss,
            ))
            total_candidate_evaluations += plain_result.candidate_evaluations

            if repeat == 0:
                key = f"n{budget}"
                primary_attention[key] = learned_attention
                primary_weights[key] = learned_weights
                primary_indices[key] = learned_result.indices
                for method, trace in (
                    ("Handcrafted-Similarity", handcrafted_result.trace),
                    ("FST-Similarity-H", learned_result.trace),
                    ("NoSimilarity-Optimized", plain_result.trace),
                ):
                    for row in trace:
                        search_rows.append({"method": method, "budget": budget, "repeat": 0, **row})

    # The complete design object is persisted and hashed before target truth is accessed.
    frozen = {
        "status": "frozen_before_target_evaluation",
        "model_sha256": train_manifest["model_sha256"],
        "candidate_hash": train_manifest["candidate_hash"],
        "designs": designs,
    }
    (run_dir / "all_frozen_designs.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    freeze_hash = sha256_file(run_dir / "all_frozen_designs.json")
    primary = {
        str(budget): next(
            design for design in designs
            if design["method"] == "FST-Similarity-H" and design["budget"] == budget and design["repeat"] == 0
        )
        for budget in test_budgets
    }
    (run_dir / "selected_sets.json").write_text(
        json.dumps({"freeze_sha256": freeze_hash, "sets": primary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "S.npz",
        **{f"S_{key}": value for key, value in primary_attention.items()},
        **{f"selected_{key}": primary_indices[key] for key in primary_indices},
    )
    weight_rows: list[dict[str, object]] = []
    for key, values in primary_weights.items():
        budget = int(key[1:])
        for position, (index, weight) in enumerate(zip(primary_indices[key], values)):
            weight_rows.append({
                "budget": budget,
                "position": position,
                "scenario_index": int(index),
                "scenario_id": identifiers[int(index)],
                "weight": float(weight),
                "source_collision_count": int(source_response[:, index].sum()),
            })
    _write_csv(run_dir / "weights.csv", weight_rows)
    _write_csv(run_dir / "search_trace.csv", search_rows)

    # Target truth is first accessed here, after every learned and baseline design is frozen.
    target_names = [str(v) for v in config["target_suts"]]
    target_rows = np.asarray([bank.index_of(name) for name in target_names], dtype=int)
    target_response = bank.collisions[target_rows].astype(float)
    target_truth = target_response @ probability
    estimate_rows: list[dict[str, object]] = []
    for design in designs:
        indices = np.asarray(design["indices"], dtype=int)
        weights = np.asarray(design["weights"], dtype=float)
        for target_name, response, truth in zip(target_names, target_response, target_truth):
            estimate_value = float(response[indices] @ weights)
            error = estimate_value - float(truth)
            estimate_rows.append({
                "method": design["method"],
                "budget": design["budget"],
                "repeat": design["repeat"],
                "target_sut": target_name,
                "estimate": estimate_value,
                "target_mu": float(truth),
                "signed_error": error,
                "absolute_error": abs(error),
                "relative_absolute_error": abs(error) / float(truth) if truth > 0 else "NA",
                "freeze_sha256": freeze_hash,
            })
    _write_csv(run_dir / "target_estimates.csv", estimate_rows)

    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in estimate_rows:
        grouped[(str(row["method"]), int(row["budget"]), str(row["target_sut"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    for (method, budget, target), rows in sorted(grouped.items()):
        estimates = np.asarray([float(row["estimate"]) for row in rows])
        errors = np.asarray([float(row["signed_error"]) for row in rows])
        absolute = np.abs(errors)
        truth = float(rows[0]["target_mu"])
        summary_rows.append({
            "method": method,
            "budget": budget,
            "target_sut": target,
            "target_mu": truth,
            "mean_absolute_error": float(absolute.mean()),
            "relative_absolute_error": float(absolute.mean() / truth) if truth > 0 else "NA",
            "signed_bias": float(errors.mean()),
            "estimate_variance": float(estimates.var(ddof=1)) if len(estimates) > 1 else 0.0,
            "absolute_error_p99": float(np.quantile(absolute, 0.99)),
            "zero_error_count": int(np.count_nonzero(absolute < 1e-12)),
            "repetitions": len(rows),
        })
    _write_csv(run_dir / "metrics_summary.csv", summary_rows)

    ideal_rows: list[dict[str, object]] = []
    ideal_rng = np.random.default_rng(seed + 777)
    for budget in test_budgets:
        indices = primary_indices[f"n{budget}"]
        weights = primary_weights[f"n{budget}"]
        source_truth = source_response @ probability
        source_estimate = source_response[:, indices] @ weights
        max_source_error = float(np.max(np.abs(source_estimate - source_truth)))
        for draw in range(64):
            coefficients = ideal_rng.dirichlet(np.ones(len(source_names)))
            synthetic = coefficients @ source_response
            error = abs(float(synthetic[indices] @ weights - synthetic @ probability))
            ideal_rows.append({
                "budget": budget,
                "draw": draw,
                "ideal_convex_target_error": error,
                "max_source_error": max_source_error,
                "bound_satisfied": bool(error <= max_source_error + 1e-10),
                "target_kind": "artificial_convex_combination_not_real_AV",
            })
    _write_csv(run_dir / "ideal_bound.csv", ideal_rows)

    audit_rows: list[dict[str, object]] = []
    for budget in test_budgets:
        indices = primary_indices[f"n{budget}"]
        attention = primary_attention[f"n{budget}"]
        for name, response in zip(source_names, source_response):
            _fluctuation, _weight, residual = signed_fluctuation(
                response, indices, attention, probability
            )
            audit_rows.append({"budget": budget, "source_sut": name, "identity_residual": residual})
    _write_csv(run_dir / "fluctuation_identity.csv", audit_rows)
    max_residual = max(abs(float(row["identity_residual"])) for row in audit_rows)
    (run_dir / "formula_audit.md").write_text(
        "# Fluctuation formula audit\n\n"
        "The reproduction keeps `fluctuation_weight: 0.0`. Direct substitution of the signed "
        "paper Eq. (24) gives `sum_i w_i F_i = mu_m - mu_hat_m`; therefore this term can "
        "duplicate the existing signed estimation error under a direct aggregation. No absolute "
        "value or variance replacement was introduced.\n\n"
        f"Maximum numerical identity residual across primary sets: `{max_residual:.3e}`.\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - design_started
    cost = {
        "source_bank_episodes_visible_to_design": int(len(source_names) * len(bank.anchors)),
        "network_training_seconds": train_manifest["training_seconds"],
        "set_design_seconds": elapsed,
        "set_optimizer_candidate_evaluations": total_candidate_evaluations,
        "primary_target_logical_queries": int(sum(test_budgets) * len(target_names)),
        "offline_target_reference_truth_episodes": int(len(target_names) * len(bank.anchors)),
        "algorithm_replays_per_method_budget": repetitions,
        "physical_simulator_cache_note": "statistical repetitions replay the deterministic bank; they are not independent AVs",
    }
    (run_dir / "cost_ledger.json").write_text(
        json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "status": "fixed_sets_evaluated",
        "freeze_sha256": freeze_hash,
        "target_truth_accessed_only_after_freeze": True,
        "budgets": test_budgets,
        "repetitions": repetitions,
        "all_ideal_bounds_satisfied": all(bool(row["bound_satisfied"]) for row in ideal_rows),
        "max_fluctuation_identity_residual": max_residual,
        "artifacts": [
            "reference_distribution.csv", "S.npz", "weights.csv", "selected_sets.json",
            "source_mu.csv", "target_estimates.csv", "search_trace.csv", "split_manifest.json",
            "metrics_summary.csv", "ideal_bound.csv", "formula_audit.md", "cost_ledger.json",
        ],
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+")
    args = parser.parse_args()
    result = run_experiment(args.run_dir, args.budgets)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
