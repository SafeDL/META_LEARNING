"""Train the FST query-normalized scenario-similarity network on source SUTs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from highway_sim_env.data.response_bank import ResponseBank

from .reference_distribution import (
    scenario_ids,
    sha256_file,
    sha256_json,
    uniform_distribution,
    write_distribution,
)
from .response_clusters import ResponseClusterSampler, sample_uniform
from .similarity_network import FeatureTransform, SimilarityNetwork, minimax_loss_per_set


def _rows(bank: ResponseBank, names: list[str]) -> np.ndarray:
    return np.asarray([bank.index_of(name) for name in names], dtype=int)


def _sample_batch(
    batch_size: int,
    n: int,
    length: int,
    rng: np.random.Generator,
    sampler: ResponseClusterSampler | None,
) -> np.ndarray:
    values = [
        sampler.sample(n, rng) if sampler is not None else sample_uniform(n, length, rng)
        for _ in range(batch_size)
    ]
    return np.stack(values)


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(config_path: Path, bank_path: Path | None, output: Path) -> dict[str, object]:
    """Train using source rows only; target rows are recorded by name but never loaded."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required_values = {
        "method": "fst_similarity_h",
        "task_kind": "performance_estimation",
        "event": "collision",
        "reference_distribution": "uniform_benchmark",
        "similarity": "inverse_l2",
        "set_optimizer": "discrete_single_swap",
        "fluctuation_weight": 0.0,
    }
    for name, expected in required_values.items():
        if config.get(name) != expected:
            raise ValueError(f"{name} must be {expected!r}")
    resolved_bank = Path(bank_path or config["bank"])
    bank = ResponseBank.load(resolved_bank)
    train_names = [str(v) for v in config["source_train_suts"]]
    dev_names = [str(v) for v in config["source_dev_suts"]]
    target_names = [str(v) for v in config["target_suts"]]
    all_names = train_names + dev_names + target_names
    if len(set(all_names)) != len(all_names):
        raise ValueError("source, development, and target SUT splits must be disjoint")
    # This is the only response read in this function. Held-out target indices are
    # deliberately not constructed, so target truth cannot enter training or stopping.
    train_response = bank.collisions[_rows(bank, train_names)].astype(np.float32)
    dev_response = bank.collisions[_rows(bank, dev_names)].astype(np.float32)
    modes = (
        np.asarray(bank.modes).astype(str)
        if bank.modes is not None
        else np.full(len(bank.anchors), "fast_intrusion", dtype="U32")
    )
    identifiers = scenario_ids(bank.anchors, modes)
    probability = uniform_distribution(len(bank.anchors))
    transform = FeatureTransform.fit(bank.anchors, modes)
    features = transform.transform(bank.anchors, modes)

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SimilarityNetwork(
        input_dim=features.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        linear_layers=int(config["encoder_layers"]),
        distance_epsilon=float(config["distance_epsilon"]),
        temperature=float(config["query_softmax_temperature"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    feature_tensor = torch.as_tensor(features)
    probability_tensor = torch.as_tensor(probability, dtype=torch.float32)
    train_tensor = torch.as_tensor(train_response)
    dev_tensor = torch.as_tensor(dev_response)
    cluster_sampler = None
    if config["pc"] == "response_cluster":
        cluster_sampler = ResponseClusterSampler.fit(
            train_response, int(config["response_clusters"]), seed
        )
    elif config["pc"] != "uniform":
        raise ValueError("pc must be response_cluster or uniform")

    train_n = int(config["train_n"])
    dev_indices = _sample_batch(64, train_n, len(features), rng, cluster_sampler)
    dev_index_tensor = torch.as_tensor(dev_indices, dtype=torch.long)
    history: list[dict[str, float | int]] = []
    best_dev = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    max_updates = int(config["max_training_updates"])
    interval = int(config["evaluation_interval"])
    for update in range(1, max_updates + 1):
        model.train()
        indices = _sample_batch(
            int(config["batch_size"]), train_n, len(features), rng, cluster_sampler
        )
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        _embedding, _attention, weights = model(
            feature_tensor, index_tensor, probability_tensor
        )
        losses, _estimates = minimax_loss_per_set(
            weights, train_tensor, index_tensor, probability_tensor
        )
        loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("similarity network produced a non-finite or zero gradient")
        optimizer.step()
        if update == 1 or update % interval == 0 or update == max_updates:
            model.eval()
            with torch.no_grad():
                _embedding, dev_attention, dev_weights = model(
                    feature_tensor, dev_index_tensor, probability_tensor
                )
                dev_loss, _ = minimax_loss_per_set(
                    dev_weights, dev_tensor, dev_index_tensor, probability_tensor
                )
                entropy = -(dev_attention.clamp_min(1e-30).log() * dev_attention).sum(1).mean()
            dev_value = float(dev_loss.mean())
            history.append({
                "update": update,
                "train_loss": float(loss.detach()),
                "dev_loss": dev_value,
                "gradient_norm": gradient_norm,
                "attention_entropy": float(entropy),
                "elapsed_seconds": time.perf_counter() - started,
            })
            if dev_value < best_dev - 1e-8:
                best_dev = dev_value
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    output.mkdir(parents=True, exist_ok=True)
    write_distribution(output / "reference_distribution.csv", identifiers, probability)
    _write_history(output / "training_history.csv", history)
    with (output / "source_mu.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "sut", "mu"])
        for split, names, response in (
            ("train", train_names, train_response),
            ("development", dev_names, dev_response),
        ):
            for name, row in zip(names, response):
                writer.writerow([split, name, f"{float(row @ probability):.17g}"])
    with (output / "candidate_table.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "scenario_index", "scenario_id", "configured_gap", "relative_speed", "mode",
            *[f"collision_{name}" for name in train_names + dev_names],
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        visible = np.vstack([train_response, dev_response])
        for index, (identifier, anchor, mode) in enumerate(zip(identifiers, bank.anchors, modes)):
            row: dict[str, object] = {
                "scenario_index": index,
                "scenario_id": identifier,
                "configured_gap": float(anchor[0]),
                "relative_speed": float(anchor[1]),
                "mode": str(mode),
            }
            for name, response in zip(train_names + dev_names, visible):
                row[f"collision_{name}"] = int(response[index])
            writer.writerow(row)
    split_manifest = {
        "source_train_suts": train_names,
        "source_dev_suts": dev_names,
        "target_suts": target_names,
        "target_truth_accessed_during_training": False,
        "protocol": "transductive finite candidate pool; input coordinates and p are visible",
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model.model_config,
        "feature_transform": transform.to_dict(),
    }
    torch.save(checkpoint, output / "similarity_model.pt")
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    candidate_records = [
        {"scenario_id": identifier, "gap": float(anchor[0]), "relative_speed": float(anchor[1]), "mode": str(mode)}
        for identifier, anchor, mode in zip(identifiers, bank.anchors, modes)
    ]
    manifest = {
        "status": "similarity_network_trained",
        "method": "FST-Similarity-H",
        "paper_method_difference": "continuous scenario gradient descent is adapted to discrete single-swap optimization",
        "bank_path": str(resolved_bank.resolve()),
        "bank_sha256": sha256_file(resolved_bank),
        "candidate_hash": sha256_json(candidate_records),
        "source_profile_hash": sha256_json(train_names + dev_names),
        "env_hash": sha256_file(Path("highway_sim_env/envs/cutin_env.py")),
        "oracle_hash": sha256_file(Path("highway_sim_env/envs/cutin_env.py")),
        "response_bank_code_hash": sha256_file(Path("highway_sim_env/data/response_bank.py")),
        "sut_profile_code_hash": sha256_file(Path("sut_algorithms/highway_env/idm_profiles.py")),
        "upstream_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "reference_distribution": "uniform_benchmark_distribution",
        "reference_semantics": "uniform mean collision rate over the frozen candidate pool; not an NDE rate",
        "event": "collision",
        "candidate_count": len(bank.anchors),
        "model_sha256": sha256_file(output / "similarity_model.pt"),
        "best_dev_loss": best_dev,
        "training_updates": history[-1]["update"],
        "training_seconds": time.perf_counter() - started,
        "seed": seed,
        "target_truth_accessed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = train(args.config, args.bank, args.output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
