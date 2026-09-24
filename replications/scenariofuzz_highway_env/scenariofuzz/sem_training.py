"""Generate/load actual source history and train the graph SEM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

from sut_algorithms.highway_env.idm_profiles import get_profile

from .corpus import ScenarioSpec, build_default_corpus, save_corpus
from .execution import execute_scenario
from .graph_builder import build_graph, stack_graphs
from .io_utils import file_hash, load_config, write_csv
from .sem_model import ScenarioEvaluationModel


def generate_source_history(config: dict, output: Path) -> dict[str, np.ndarray]:
    """Create independent, genuine source-SUT episodes; no rows are replicated."""
    count = int(config["source_scenarios"])
    sampler = qmc.Sobol(d=2, scramble=True, seed=int(config["random_seed"]))
    unit = sampler.random_base2(int(np.ceil(np.log2(count))))[:count]
    bounds = config["bounds"]
    points = qmc.scale(unit, np.asarray([bounds["initial_gap"][0], bounds["relative_speed"][0]]), np.asarray([bounds["initial_gap"][1], bounds["relative_speed"][1]]))
    modes = tuple(config["mode_set"])
    specs = [ScenarioSpec.create(point[0], point[1], modes[i % len(modes)]) for i, point in enumerate(points)]
    rows: dict[str, list] = {key: [] for key in ("scenario_id", "initial_gap", "relative_speed", "mode", "sut_name", "collision", "near_miss", "vulnerability", "valid", "wall_seconds")}
    for sut_offset, sut_name in enumerate(config["source_suts"]):
        profile = get_profile(sut_name)
        for i, spec in enumerate(specs):
            observation = execute_scenario(profile, spec, int(config["random_seed"]) + 10000 * sut_offset + i)
            rows["scenario_id"].append(spec.scenario_id)
            rows["initial_gap"].append(spec.initial_gap)
            rows["relative_speed"].append(spec.relative_speed)
            rows["mode"].append(spec.mode)
            rows["sut_name"].append(sut_name)
            rows["collision"].append(observation.collision)
            rows["near_miss"].append(observation.near_miss)
            rows["vulnerability"].append(observation.vulnerability)
            rows["valid"].append(observation.valid)
            rows["wall_seconds"].append(observation.wall_seconds)
    data = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(output, **data)
    return data


def load_source_history(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"scenario_id", "initial_gap", "relative_speed", "mode", "sut_name", "collision", "valid"}
        missing = required - set(loaded.files)
        if missing:
            raise ValueError(f"source bank is not a ScenarioFuzz-H history bank; missing {sorted(missing)}")
        return {key: loaded[key] for key in loaded.files}


def _split_by_scenario(data: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    unique = np.unique(data["scenario_id"].astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    audit_n = max(1, int(round(0.20 * len(unique))))
    dev_n = max(1, int(round(0.16 * len(unique))))
    audit, dev, train = set(unique[:audit_n]), set(unique[audit_n:audit_n + dev_n]), set(unique[audit_n + dev_n:])
    ids = data["scenario_id"].astype(str)
    return {
        "train": np.asarray([value in train for value in ids]),
        "dev": np.asarray([value in dev for value in ids]),
        "audit": np.asarray([value in audit for value in ids]),
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "positive_fraction": float(np.mean(labels)),
    }


def _train_model(config: dict, tensors: dict[str, torch.Tensor], labels: torch.Tensor, train_indices: np.ndarray, dev_indices: np.ndarray, device: str, history_tag: str) -> tuple[ScenarioEvaluationModel, list[dict], int]:
    training_seed = int(config["random_seed"]) + int(history_tag)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    kwargs = {"hidden": int(config["hidden"]), "heads": int(config["heads"]), "dropout": float(config["dropout"])}
    model = ScenarioEvaluationModel(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    train_labels = labels[train_indices]
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    history: list[dict] = []
    max_epochs = int(config["epochs"])
    patience = int(config["early_stopping_patience"])
    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model({key: value[train_indices] for key, value in tensors.items()})
        loss = criterion(logits, train_labels)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_logits = model({key: value[dev_indices] for key, value in tensors.items()})
            dev_loss = torch.nn.functional.binary_cross_entropy_with_logits(dev_logits, labels[dev_indices]).item()
        history.append({"history_size": history_tag, "epoch": epoch, "train_loss": float(loss.item()), "dev_loss": float(dev_loss)})
        if dev_loss < best_loss - 1e-5:
            best_loss, stale, best_epoch = dev_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train(config_path: Path, source_bank: Path | None, output: Path) -> Path:
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    seeds = build_default_corpus(config)
    save_corpus(output / "seed_corpus.json", seeds)
    bank_path = source_bank or output / "source_history.npz"
    started = time.perf_counter()
    data = load_source_history(bank_path) if bank_path.exists() else generate_source_history(config, bank_path)
    # Re-apply the source/target boundary even for externally supplied banks.
    # This makes a universal physical episode bank safe to reuse in LOSO runs.
    source_suts = set(str(name) for name in config["source_suts"])
    excluded_target = str(config["target_sut"])
    if excluded_target in source_suts:
        raise ValueError("target_sut must not appear in source_suts")
    source_mask = np.asarray([str(name) in source_suts for name in data["sut_name"]])
    valid = data["valid"].astype(bool) & source_mask
    data = {key: value[valid] for key, value in data.items()}
    observed_suts = set(data["sut_name"].astype(str))
    if observed_suts != source_suts:
        missing = sorted(source_suts - observed_suts)
        unexpected = sorted(observed_suts - source_suts)
        raise ValueError(f"source bank SUT mismatch; missing={missing}, unexpected={unexpected}")
    if excluded_target in observed_suts:
        raise RuntimeError("excluded target leaked into SEM source history")
    split = _split_by_scenario(data, int(config["random_seed"]))
    # Hard assertion of scenario-level isolation.
    split_sets = {name: set(data["scenario_id"][mask].astype(str)) for name, mask in split.items()}
    if split_sets["train"] & split_sets["dev"] or split_sets["train"] & split_sets["audit"] or split_sets["dev"] & split_sets["audit"]:
        raise RuntimeError("scenario_id leakage across train/dev/audit")
    graphs = [build_graph(seeds[0], ScenarioSpec.create(g, s, str(m))) for g, s, m in zip(data["initial_gap"], data["relative_speed"], data["mode"])]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = stack_graphs(graphs, device)
    labels = torch.as_tensor(data["collision"].astype(np.float32), device=device)
    allowed = np.flatnonzero(split["train"] | split["dev"])
    rng = np.random.default_rng(int(config["random_seed"]) + 91)
    rng.shuffle(allowed)
    requested_sizes = config["history_sizes"]
    history_rows: list[dict] = []
    size_rows: list[dict] = []
    final_model = None
    final_best_epoch = 0
    for requested in requested_sizes:
        size = len(allowed) if requested == "all" else min(int(requested), len(allowed))
        selected = allowed[:size]
        train_indices = selected[split["train"][selected]]
        dev_indices = selected[split["dev"][selected]]
        if len(train_indices) < 2 or len(dev_indices) < 2:
            continue
        tag = str(size)
        model, history, best_epoch = _train_model(config, tensors, labels, train_indices, dev_indices, device, tag)
        history_rows.extend(history)
        audit_indices = np.flatnonzero(split["audit"])
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model({key: value[audit_indices] for key, value in tensors.items()})).cpu().numpy()
        metrics = _metrics(data["collision"][audit_indices].astype(int), probabilities)
        size_rows.append({"history_size": size, "best_epoch": best_epoch, **metrics})
        size_checkpoint = output / f"source_only_sem_history_{size}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "hidden": int(config["hidden"]),
                "heads": int(config["heads"]),
                "dropout": float(config["dropout"]),
            },
            "graph_schema": config["graph_schema"],
            "source_suts": sorted(source_suts),
            "target_sut_excluded": excluded_target,
            "best_epoch": best_epoch,
            "history_size": size,
        }, size_checkpoint)
        final_model, final_best_epoch = model, best_epoch
    if final_model is None:
        raise RuntimeError("no SEM model was trained")
    audit_indices = np.flatnonzero(split["audit"])
    final_model.eval()
    with torch.no_grad():
        audit_probabilities = torch.sigmoid(final_model({key: value[audit_indices] for key, value in tensors.items()})).cpu().numpy()
    audit_rows = []
    for index, probability in zip(audit_indices, audit_probabilities):
        audit_rows.append({
            "scenario_id": str(data["scenario_id"][index]), "sut_name": str(data["sut_name"][index]),
            "initial_gap": float(data["initial_gap"][index]), "relative_speed": float(data["relative_speed"][index]),
            "mode": str(data["mode"][index]), "actual_collision": int(data["collision"][index]),
            "predicted_score": float(probability), "split": "frozen_audit",
        })
    write_csv(output / "training_history.csv", history_rows)
    write_csv(output / "history_size_metrics.csv", size_rows)
    write_csv(output / "audit_predictions.csv", audit_rows)
    model_kwargs = {"hidden": int(config["hidden"]), "heads": int(config["heads"]), "dropout": float(config["dropout"])}
    checkpoint_path = output / "source_only_sem.pt"
    torch.save({"state_dict": final_model.state_dict(), "model_kwargs": model_kwargs, "graph_schema": config["graph_schema"], "source_suts": list(config["source_suts"]), "target_sut_excluded": config["target_sut"], "best_epoch": final_best_epoch}, checkpoint_path)
    manifest = {
        "method": "ScenarioFuzz-H-SEM", "source_bank": bank_path.as_posix(), "source_bank_sha256": file_hash(bank_path),
        "independent_actual_episodes": int(len(data["scenario_id"])), "unique_scenarios": int(len(np.unique(data["scenario_id"]))),
        "source_suts": list(config["source_suts"]), "excluded_target_sut": config["target_sut"],
        "split_protocol": "scenario_id_grouped_train_dev_audit", "train_schedule": config["train_schedule"],
        "configured_max_epochs": int(config["epochs"]), "actual_final_best_epoch": final_best_epoch,
        "device": device, "training_and_bank_wall_seconds": time.perf_counter() - started,
        "model_initialization_seed_rule": "config.random_seed + actual_history_size",
        "paper_deviation": "early stopping is an explicitly named low-compute adaptation; rejected candidates never enter training",
        "history_checkpoints": sorted(path.name for path in output.glob("source_only_sem_history_*.pt")),
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = train(args.config, args.source_bank, args.output)
    print(f"Wrote leakage-isolated SEM checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
