"""Post-hoc collision clustering; never feeds the online selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from diva_highway_env.sut.idm_profiles import get_profile

from .corpus import ScenarioSpec, build_default_corpus
from .filter import load_checkpoint
from .graph_builder import build_graph, stack_graphs
from .io_utils import load_config, write_csv
from .execution import execute_scenario
from .sem_training import _split_by_scenario, load_source_history
from .trajectory_encoder import learn_embeddings


def cluster(run_dir: Path, config_path: Path, sem_path: Path) -> None:
    rows = [json.loads(line) for line in (run_dir / "executions.jsonl").read_text(encoding="utf-8").splitlines()]
    unique = {}
    for row in rows:
        if row["collision"] and row["trajectory_path"]:
            unique.setdefault(row["scenario_id"], row)
    collision_rows = list(unique.values())
    if len(collision_rows) < 4:
        (run_dir / "clustering_status.json").write_text(json.dumps({"status": "insufficient_clustering_data", "unique_collision_trajectories": len(collision_rows), "minimum": 4}, indent=2), encoding="utf-8")
        write_csv(run_dir / "clusters.csv", [])
        return
    config = load_config(config_path); seed = build_default_corpus(config)[0]
    model_dir = sem_path.resolve().parent
    training_manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    source = load_source_history(Path(training_manifest["source_bank"]))
    source_split = _split_by_scenario(source, int(config["random_seed"]))
    dev_indices = np.flatnonzero(source_split["dev"] & source["collision"].astype(bool) & source["valid"].astype(bool))
    # A fixed source-development subset selects clustering hyperparameters.
    dev_rows = []
    dev_dir = model_dir / "cluster_dev_trajectories"
    for replay_number, index in enumerate(dev_indices[:32]):
        spec = ScenarioSpec.create(source["initial_gap"][index], source["relative_speed"][index], str(source["mode"][index]))
        observation = execute_scenario(get_profile(str(source["sut_name"][index])), spec, int(config["random_seed"]) + 700000 + replay_number, dev_dir)
        if observation.collision and observation.trajectory_path:
            dev_rows.append({"spec": spec, "sut_name": str(source["sut_name"][index]), "trajectory_path": observation.trajectory_path})
    if len(dev_rows) < 4:
        raise RuntimeError("source development set has insufficient collision trajectories for clustering selection")
    all_paths = [row["trajectory_path"] for row in dev_rows] + [row["trajectory_path"] for row in collision_rows]
    all_trajectory_embeddings = learn_embeddings(all_paths, int(config["random_seed"]))
    dev_count = len(dev_rows)
    model, _ = load_checkpoint(sem_path)
    dev_graphs = [build_graph(seed, row["spec"]) for row in dev_rows]
    target_graphs = [build_graph(seed, ScenarioSpec.create(row["initial_gap"], row["relative_speed"], row["mode"])) for row in collision_rows]
    with torch.no_grad():
        all_sem_embeddings = model.embedding(stack_graphs(dev_graphs + target_graphs)).numpy()
    raw = np.concatenate((all_trajectory_embeddings, all_sem_embeddings), axis=1)
    scaler = StandardScaler().fit(raw[:dev_count])
    dev_combined = scaler.transform(raw[:dev_count]); combined = scaler.transform(raw[dev_count:])
    dev_distances = np.linalg.norm(dev_combined[:, None] - dev_combined[None, :], axis=-1)
    pairwise = dev_distances[np.triu_indices(len(dev_combined), 1)]
    candidates = np.unique(np.quantile(pairwise, np.linspace(.25, .75, 11)))
    threshold, best_silhouette = None, -1.0
    for candidate in candidates:
        dev_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=max(float(candidate), 1e-6), linkage="average").fit_predict(dev_combined)
        if 1 < len(np.unique(dev_labels)) < len(dev_labels):
            score = float(silhouette_score(dev_combined, dev_labels))
            if score > best_silhouette:
                threshold, best_silhouette = float(candidate), score
    if threshold is None:
        raise RuntimeError("development trajectories could not select a non-degenerate clustering threshold")
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage="average").fit_predict(combined)
    coordinates = PCA(n_components=2, random_state=int(config["random_seed"])).fit_transform(combined)
    cluster_rows = []
    for row, label, xy in zip(collision_rows, labels, coordinates):
        cluster_rows.append({"scenario_id": row["scenario_id"], "execution_id": row["execution_id"], "mode": row["mode"], "cluster": int(label), "pca_x": float(xy[0]), "pca_y": float(xy[1]), "trajectory_path": row["trajectory_path"]})
    write_csv(run_dir / "clusters.csv", cluster_rows)
    np.savez_compressed(run_dir / "cluster_embeddings.npz", combined=combined, trajectory=all_trajectory_embeddings[dev_count:], sem=all_sem_embeddings[dev_count:], labels=labels)
    (run_dir / "clustering_status.json").write_text(json.dumps({"status": "completed", "unique_collision_trajectories": len(collision_rows), "clusters": int(len(np.unique(labels))), "algorithm": "masked-transformer+SEM / agglomerative-average", "distance_threshold": threshold, "threshold_selected_on": "source_development_collision_trajectories", "source_development_replays": len(dev_rows), "development_silhouette": best_silhouette}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sem", type=Path, required=True)
    args = parser.parse_args(); cluster(args.run_dir, args.config, args.sem)
    print(f"Wrote post-analysis to {args.run_dir}")


if __name__ == "__main__":
    main()
