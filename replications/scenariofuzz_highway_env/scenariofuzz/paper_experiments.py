"""Run the paper-aligned multi-system ScenarioFuzz-H experiment suite.

The suite keeps the highway-env adaptation honest: every target system gets a
separate LOSO SEM, target audit labels never enter training or selection, and
all discovery curves come from new physical highway-env executions.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from diva_highway_env.sut.idm_profiles import PROFILE_NAMES

from .corpus import ScenarioSpec, build_default_corpus
from .filter import load_checkpoint, predict_scores
from .fuzz_campaign import run_campaign
from .io_utils import file_hash, load_config, write_csv
from .paper_figures import build_paper_aligned_outputs
from .sem_training import (
    _split_by_scenario,
    generate_source_history,
    load_source_history,
    train,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _flat_features(data: dict[str, np.ndarray]) -> np.ndarray:
    mode_names = ("fast_intrusion", "cutin_braking", "lead_braking")
    one_hot = np.asarray([[float(str(mode) == name) for name in mode_names] for mode in data["mode"]])
    return np.column_stack((data["initial_gap"].astype(float), data["relative_speed"].astype(float), one_hot))


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray, seed: int) -> dict[str, float]:
    labels = labels.astype(int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predicted = probabilities >= 0.5
    correlation = float(pearsonr(labels, probabilities).statistic) if len(np.unique(labels)) > 1 else float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    hit_values: list[float] = []
    for start in range(0, len(order), 16):
        batch = order[start:start + 16]
        if len(batch) < 3 or not labels[batch].any():
            continue
        selected = batch[np.argsort(probabilities[batch])[::-1][:3]]
        hit_values.append(float(labels[selected].any()))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "cross_entropy": float(log_loss(labels, probabilities, labels=[0, 1])),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "pearson": correlation,
        "positive_fraction": float(labels.mean()),
        "top3_hit_rate_in_positive_16_candidate_batches": float(np.mean(hit_values)) if hit_values else float("nan"),
        "top3_evaluable_batches": len(hit_values),
    }


def _target_audit(
    universal: dict[str, np.ndarray],
    config: dict,
    checkpoint_path: Path,
    output: Path,
) -> list[dict]:
    source_suts = set(str(name) for name in config["source_suts"])
    source_mask = universal["valid"].astype(bool) & np.asarray(
        [str(name) in source_suts for name in universal["sut_name"]]
    )
    source = {key: value[source_mask] for key, value in universal.items()}
    split = _split_by_scenario(source, int(config["random_seed"]))
    audit_ids = set(source["scenario_id"][split["audit"]].astype(str))
    target_mask = universal["valid"].astype(bool) & np.asarray(
        [str(name) == config["target_sut"] and str(sid) in audit_ids
         for name, sid in zip(universal["sut_name"], universal["scenario_id"])]
    )
    target = {key: value[target_mask] for key, value in universal.items()}
    if len(target["scenario_id"]) != len(audit_ids):
        raise RuntimeError("target audit does not contain exactly one row per held-out scenario")

    specs = [
        ScenarioSpec.create(gap, speed, str(mode))
        for gap, speed, mode in zip(target["initial_gap"], target["relative_speed"], target["mode"])
    ]
    model, _ = load_checkpoint(checkpoint_path, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device.type
    sem_probabilities = predict_scores(model, build_default_corpus(config)[0], specs, device)
    labels = target["collision"].astype(int)

    train_mask = ~np.asarray([str(sid) in audit_ids for sid in source["scenario_id"]])
    train_data = {key: value[train_mask] for key, value in source.items()}
    x_train, y_train = _flat_features(train_data), train_data["collision"].astype(int)
    x_test = _flat_features(target)
    seed = int(config["random_seed"])
    baselines = {
        "RBF-Filter-H": make_pipeline(
            StandardScaler(), RBFSampler(gamma=0.35, n_components=96, random_state=seed),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "MLP-SEM-H": make_pipeline(
            StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=600, early_stopping=True,
                random_state=seed, learning_rate_init=1e-3,
            ),
        ),
        "RandomForest-Filter-H": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        ),
    }
    probabilities: dict[str, np.ndarray] = {"Graph-SEM-H": sem_probabilities}
    for name, estimator in baselines.items():
        estimator.fit(x_train, y_train)
        probabilities[name] = estimator.predict_proba(x_test)[:, 1]

    prediction_rows: list[dict] = []
    for index, spec in enumerate(specs):
        prediction_rows.append({
            "target_sut": config["target_sut"],
            "scenario_id": spec.scenario_id,
            "initial_gap": spec.initial_gap,
            "relative_speed": spec.relative_speed,
            "mode": spec.mode,
            "actual_collision": int(labels[index]),
            **{f"{name}_probability": float(values[index]) for name, values in probabilities.items()},
        })
    write_csv(output / "target_audit_predictions.csv", prediction_rows)

    rows: list[dict] = []
    for name, values in probabilities.items():
        rows.append({
            "target_sut": config["target_sut"],
            "model": name,
            "audit_records": len(labels),
            "audit_scenarios_unseen_during_training": True,
            "target_sut_excluded_from_training": True,
            **_probability_metrics(labels, values, seed + len(name)),
        })
    write_csv(output / "target_audit_metrics.csv", rows)
    return rows


def _campaign_complete(path: Path, expected_executions: int) -> bool:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8"))["total_actual_target_executions"]) == expected_executions
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def run_suite(pool_config_path: Path, online_config_path: Path, output: Path, repeats: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pool_base = load_config(pool_config_path)
    online_base = load_config(online_config_path)
    base_seed = int(pool_base["random_seed"])

    universal_path = output / "universal_history.npz"
    universal_config = deepcopy(pool_base)
    universal_config["source_suts"] = list(PROFILE_NAMES)
    universal_config["target_sut"] = "LOSO_HELD_OUT_PER_MODEL"
    _write_yaml(output / "universal_history_config.yaml", universal_config)
    if not universal_path.exists():
        generate_source_history(universal_config, universal_path)
    universal = load_source_history(universal_path)
    expected_universal = len(PROFILE_NAMES) * int(pool_base["source_scenarios"])
    if len(universal["scenario_id"]) != expected_universal:
        raise RuntimeError(f"universal history has {len(universal['scenario_id'])}, expected {expected_universal}")

    all_model_metrics: list[dict] = []
    campaign_entries: list[dict] = []
    campaign_summary_rows: list[dict] = []
    final_checkpoints: dict[str, Path] = {}
    for target in PROFILE_NAMES:
        model_dir = output / "models" / target
        model_config = deepcopy(pool_base)
        model_config["source_suts"] = [name for name in PROFILE_NAMES if name != target]
        model_config["target_sut"] = target
        model_config["random_seed"] = base_seed
        model_config["history_sizes"] = [100, 300, "all"] if target == "SUT-C" else ["all"]
        model_config_path = model_dir / "config.resolved.yaml"
        _write_yaml(model_config_path, model_config)
        checkpoint = model_dir / "source_only_sem.pt"
        if not checkpoint.exists():
            checkpoint = train(model_config_path, universal_path, model_dir)
        final_checkpoints[target] = checkpoint
        all_model_metrics.extend(_target_audit(universal, model_config, checkpoint, model_dir))

        for repeat in range(repeats):
            random_seed = base_seed + repeat
            run_config = deepcopy(online_base)
            run_config["target_sut"] = target
            run_config["random_seed"] = random_seed
            run_dir = output / "campaigns" / target / f"seed_{random_seed}"
            config_path = run_dir / "suite_input.yaml"
            _write_yaml(config_path, run_config)
            if not _campaign_complete(run_dir, int(run_config["budget"]) * len(run_config["variants"]) * len(run_config["protocols"])):
                run_campaign(config_path, checkpoint, run_dir)
            for row in _read_csv(run_dir / "ablation_summary.csv"):
                campaign_summary_rows.append({"repeat": repeat, "random_seed": random_seed, **row})
            campaign_entries.append({"target_sut": target, "repeat": repeat, "random_seed": random_seed, "run_dir": run_dir.as_posix()})

    write_csv(output / "filter_model_metrics.csv", all_model_metrics)
    write_csv(output / "multi_system_runs.csv", campaign_summary_rows)

    history_entries: list[dict] = []
    history_summary_rows: list[dict] = []
    c_model_dir = output / "models" / "SUT-C"
    for checkpoint in sorted(c_model_dir.glob("source_only_sem_history_*.pt")):
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        history_size = int(metadata["history_size"])
        for repeat in range(repeats):
            random_seed = base_seed + repeat
            run_config = deepcopy(online_base)
            run_config["target_sut"] = "SUT-C"
            run_config["random_seed"] = random_seed
            run_config["variants"] = ["2SMS+SEM-H"]
            run_config["protocols"] = ["paper_nm"]
            run_dir = output / "history_sweep" / f"history_{history_size}" / f"seed_{random_seed}"
            config_path = run_dir / "suite_input.yaml"
            _write_yaml(config_path, run_config)
            if not _campaign_complete(run_dir, int(run_config["budget"])):
                run_campaign(config_path, checkpoint, run_dir)
            row = _read_csv(run_dir / "ablation_summary.csv")[0]
            history_summary_rows.append({"history_size": history_size, "repeat": repeat, "random_seed": random_seed, **row})
            history_entries.append({"history_size": history_size, "repeat": repeat, "random_seed": random_seed, "run_dir": run_dir.as_posix()})
    write_csv(output / "history_discovery_runs.csv", history_summary_rows)

    manifest = {
        "suite": "ScenarioFuzz-H-paper-aligned",
        "paper_scope": "highway-env behavioral adaptation; not CARLA numeric reproduction",
        "source_pdf": "Dance of the ADS: Orchestrating Failures through Historically-Informed Scenario Fuzzing",
        "targets": list(PROFILE_NAMES),
        "algorithm_repeats": repeats,
        "base_random_seed": base_seed,
        "universal_history": universal_path.as_posix(),
        "universal_history_sha256": file_hash(universal_path),
        "universal_actual_episodes": expected_universal,
        "source_episodes_per_loso_model": (len(PROFILE_NAMES) - 1) * int(pool_base["source_scenarios"]),
        "target_audit_records_per_sut": max(1, int(round(pool_base["source_scenarios"] * 0.20))),
        "campaigns": campaign_entries,
        "history_sweep": history_entries,
        "target_truth_used_for_training_or_selection": False,
        "weather_color_perception_claims": False,
        "code_coverage_claimed": False,
    }
    (output / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_paper_aligned_outputs(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-config", type=Path, required=True)
    parser.add_argument("--online-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("paper-aligned suite requires at least two algorithm repeats")
    run_suite(args.pool_config, args.online_config, args.output, args.repeats)
    print(f"Wrote paper-aligned ScenarioFuzz-H suite to {args.output}")


if __name__ == "__main__":
    main()
