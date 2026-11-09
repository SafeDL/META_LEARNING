"""Completed paid tasks, bounded source views and between-task model publication."""

import gzip
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from testing_methods.common.response import ResponseArchive, archive_fingerprints, source_weights
from testing_methods.history_transfer.inference import historical_prior

from .rss import measurement_contract, trajectory_margin
from .scenarios import features
from .schema import QueryFeedback, Scenario
from .storage import load_archive, read_json, write_json


def model_version(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def finalize_task(
    destination,
    ledger,
    *,
    task_id,
    sut_id,
    sut_version,
    family,
    lineage,
    environment_seed,
    completed_order,
    branch_id,
    original_role="target",
    execution_contract=None,
    model_snapshot=None,
):
    """Archive only ledger-backed successful executions; retain every failed attempt."""
    if not ledger.records or any(r["status"] == "running" for r in ledger.records):
        raise ValueError("Only ended tasks with metered attempts can be finalized")
    successful = [r for r in ledger.records if r["status"] == "complete"]
    if len(successful) != len(ledger.trajectories):
        raise ValueError("Successful attempts and evaluator trajectories must align")
    executions = []
    for record, trajectory in zip(successful, ledger.trajectories):
        scene = ledger.candidates[record["index"]]
        feedback = QueryFeedback.from_dict(record["feedback"])
        if (
            scene.scenario_id != feedback.scenario_id
            or bool(any(f["collision"] for f in trajectory)) != feedback.collision
        ):
            raise ValueError("Archive response must match its paid scene and actual collision")
        margin = trajectory_margin(trajectory)["minimum_margin"]
        if feedback.safety.rss_margin is not None and not np.isclose(
            feedback.safety.rss_margin, margin, atol=1e-12
        ):
            raise ValueError("Paid RSS differs from its executed trajectory")
        feedback = replace(
            feedback,
            safety=replace(feedback.safety, rss_margin=margin),
            rss_contract=measurement_contract(),
        )
        executions.append(
            {
                "record_id": f"{task_id}:{record['call']}",
                "query": record["call"],
                "scenario": asdict(scene),
                "feedback": feedback.to_dict(),
                "trajectory": trajectory,
            }
        )
    payload = {
        "format": "completed-paid-task",
        "completed": True,
        "task_id": task_id,
        "sut_id": sut_id,
        "sut_version": sut_version,
        "family": family,
        "lineage": lineage,
        "original_role": original_role,
        "execution_contract": execution_contract,
        "model_snapshot": model_snapshot,
        "environment_seed": environment_seed,
        "completed_order": completed_order,
        "branch_id": branch_id,
        "rss_contract": measurement_contract(),
        "attempts": ledger.records,
        "executions": executions,
    }
    payload = json.loads(json.dumps(payload, allow_nan=False))
    destination = Path(destination)
    if destination.exists():
        if json.loads(gzip.decompress(destination.read_bytes())) != payload:
            raise ValueError("A completed task identifier cannot be reused with different records")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(
        gzip.compress(json.dumps(payload, allow_nan=False).encode("utf-8"), mtime=0)
    )
    temporary.replace(destination)
    return destination


def completed_archives(paths, *, before_order, branch_id, current_system=None, held_family=None):
    """Admit past tasks in this branch, retaining target origin and ADS version."""
    groups = {}
    tasks = [json.loads(gzip.decompress(Path(path).read_bytes())) for path in paths]
    for task in sorted(tasks, key=lambda t: (t["completed_order"], t["task_id"])):
        if task["format"] != "completed-paid-task" or not task["completed"]:
            raise ValueError("Not a completed paid task")
        if (
            task["branch_id"] != branch_id
            or task["completed_order"] >= before_order
            or (task["sut_id"], task["sut_version"]) == current_system
            or (held_family is not None and task["family"] == held_family)
        ):
            continue
        if task["original_role"] == "development":
            continue
        if task["rss_contract"] != measurement_contract():
            raise ValueError("Completed source uses another RSS measure")
        paid = {r["call"]: r for r in task["attempts"] if r["status"] == "complete"}
        for execution in task["executions"]:
            scene = Scenario(**execution["scenario"])
            feedback = QueryFeedback.from_dict(execution["feedback"])
            record = paid[execution["query"]]
            if (
                record["scenario_id"] != scene.scenario_id
                or record["feedback"]["collision"] != feedback.collision
            ):
                raise ValueError("Completed source lacks matching paid execution evidence")
            key = task["sut_id"], task["sut_version"], scene.kind
            group = groups.setdefault(key, {"task": task, "records": {}, "task_ids": []})
            if group["task"]["execution_contract"] != task["execution_contract"]:
                raise ValueError("Different execution contracts require distinct system versions")
            if task["task_id"] not in group["task_ids"]:
                group["task_ids"].append(task["task_id"])
            # Repetitions stay in the raw task; the deterministic model view uses the first.
            group["records"].setdefault(scene.coordinate(), (execution, scene, feedback))
    archives = []
    for (sut_id, version, kind), group in groups.items():
        task = group["task"]
        records = list(group["records"].values())
        archives.append(
            ResponseArchive(
                f"{sut_id}@{version}:{kind}",
                task["family"],
                task["lineage"],
                "target",
                [s for _, s, _ in records],
                np.array([f.collision for _, _, f in records], dtype=bool),
                np.array([[f.safety.min_clearance_m, f.safety.min_ttc_s] for _, _, f in records]),
                provenance={
                    "completed_task": True,
                    "original_role": task["original_role"],
                    "contract": task["execution_contract"],
                    "environment_seed": task["environment_seed"],
                    "sut_id": sut_id,
                    "sut_version": version,
                    "branch_id": branch_id,
                    "task_ids": group["task_ids"],
                    "record_ids": [r["record_id"] for r, _, _ in records],
                },
                rss_margin=np.array([f.safety.rss_margin for _, _, f in records]),
                rss_contract=measurement_contract(),
            )
        )
    return archives


def calculation_view(archive, *, limit=512, seed=1301, use_rss=True, record_ids=None):
    """Select a bounded view, or replay an explicitly registered record sequence."""
    if limit < 1:
        raise ValueError("Source view limit must be positive")
    n = len(archive.scenarios)
    if record_ids is not None:
        available = archive.provenance.get("record_ids", [])
        if len(available) != n or len(set(available)) != n:
            raise ValueError("Fixed source views require unique archive record IDs")
        if not record_ids or len(record_ids) > limit or len(set(record_ids)) != len(record_ids):
            raise ValueError("Fixed source record IDs must be nonempty, unique and within limit")
        positions = {name: i for i, name in enumerate(available)}
        if any(name not in positions for name in record_ids):
            raise ValueError("Fixed source record ID is absent from the archive")
        indices = np.array([positions[name] for name in record_ids], dtype=int)
    elif n <= limit:
        indices = np.arange(n)
    else:
        y = archive.collision
        if use_rss:
            r = archive.continuous_margin
            strata = [
                np.flatnonzero(y),
                np.flatnonzero(~y & (np.abs(r) <= 0.1)),
                np.flatnonzero(~y & (r < -0.1)),
                np.flatnonzero(~y & (r > 0.1)),
            ]
        else:
            strata = [np.flatnonzero(y), np.flatnonzero(~y)]
        generator = np.random.default_rng(seed)
        strata = [list(generator.permutation(s)) for s in strata]
        chosen = []
        while len(chosen) < limit:
            for stratum in strata:
                if stratum and len(chosen) < limit:
                    chosen.append(stratum.pop())
        indices = np.sort(chosen)
    provenance = {**archive.provenance, "calculation_indices": indices.tolist()}
    if "record_ids" in provenance:
        provenance["record_ids"] = [archive.provenance["record_ids"][i] for i in indices]
    return replace(
        archive,
        scenarios=[archive.scenarios[i] for i in indices],
        collision=archive.collision[indices],
        safety=archive.safety[indices],
        rss_margin=archive.continuous_margin[indices],
        provenance=provenance,
    )


def build_source(
    model,
    archives,
    candidates,
    *,
    cache_path=None,
    cache_directory=None,
    limit=512,
    batch_size=128,
    use_rss=True,
    view_record_ids=None,
):
    kind = candidates[0].kind
    if any(s.kind != kind for s in candidates) or any(
        a.scenarios[0].kind != kind for a in archives
    ):
        raise ValueError("Historical sources and candidates must share one functional scenario")
    construction = getattr(model, "construction_specification", None)
    if construction is not None and construction["scenario"] != kind:
        raise ValueError("Shared model belongs to another functional scenario")
    if construction is not None and construction["config"]["historical_rss"] != use_rss:
        raise ValueError("Source feedback differs from shared model construction")
    if view_record_ids is not None and set(view_record_ids) != {a.source_id for a in archives}:
        raise ValueError("Fixed views must identify every admitted source exactly once")
    views = [
        calculation_view(
            a,
            limit=limit,
            use_rss=use_rss,
            record_ids=None if view_record_ids is None else view_record_ids[a.source_id],
        )
        for a in archives
    ]
    spec = {
        "format": "joint-source-blocks",
        "scenario": kind,
        "model_version": model_version(model),
        "archives": archive_fingerprints(archives),
        "rss_contract": measurement_contract(),
        "historical_rss": use_rss,
        "views": {a.source_id: a.provenance["calculation_indices"] for a in views},
        "record_ids": {a.source_id: a.provenance.get("record_ids", []) for a in archives},
        "view_record_ids": {a.source_id: a.provenance.get("record_ids", []) for a in views},
        "candidates": [asdict(s) for s in candidates],
        "covariance_mode": "point_joint_blocks",
    }
    if cache_directory is not None:
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        cache_path = Path(cache_directory) / f"{key}.npz"
    if cache_path is not None and Path(cache_path).exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if json.loads(str(saved["specification"])) != spec:
                raise ValueError(
                    "Source cache differs from model, archive, RSS or candidate snapshot"
                )
            return {
                "means": saved["means"].copy(),
                "blocks": saved["blocks"].copy(),
                "weights": saved["weights"].copy(),
                "names": saved["names"].tolist(),
                "specification": spec,
            }
    query_x = features(candidates)
    means, blocks = [], []
    with torch.no_grad():
        encoded_query = model.features(query_x)
        for archive in views:
            mean, covariance = historical_prior(
                model,
                archive,
                query_x,
                batch_size=batch_size,
                encoded_query=encoded_query,
                use_rss=use_rss,
            )
            means.append(mean.cpu().numpy())
            blocks.append(covariance.cpu().numpy())
    result = {
        "means": np.stack(means),
        "blocks": np.stack(blocks),
        "weights": source_weights(views),
        "names": [a.source_id for a in views],
        "specification": spec,
    }
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **{k: result[k] for k in ("means", "blocks", "weights")},
                names=np.asarray(result["names"]),
                specification=json.dumps(spec),
            )
        temporary.replace(path)
    return result


def load_training_archives(manifest_path, kind, held_family=None):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    history = []
    for entry in manifest["archives"]:
        if entry["scenario"] != kind:
            continue
        archive = load_archive(manifest_path.parent / entry["path"])
        if archive.role != "history" or archive.source_id != entry["source_id"]:
            raise ValueError("Construction manifest differs from archive identity")
        if archive.scenarios[0].kind != kind:
            raise ValueError("Construction archive belongs to another functional scenario")
        if archive_fingerprints([archive])[archive.source_id] != entry["fingerprint"]:
            raise ValueError("Construction archive changed after registration")
        if held_family is None or archive.family != held_family:
            archive.continuous_margin
            history.append(archive)
    if not history or len({a.source_id for a in history}) != len(history):
        raise ValueError("Construction requires nonempty, distinct historical sources")
    return history


def update_shared_model(destination, history, current_models, *, new_source_ids, config=None):
    """Publish the fixed-step candidate between tasks if numerical validation succeeds."""
    from testing_methods.history_transfer.training import DEFAULT, load_model, specification, train

    root = Path(destination)
    parent = model_version(current_models[0]) + ":" + model_version(current_models[1])
    config = config or DEFAULT
    spec = specification(history, None, config)
    spec.update(new_source_ids=list(new_source_ids), parent_version=parent)
    publication_path = root / "publication.json"
    if publication_path.exists():
        existing = read_json(publication_path)
        if existing["specification"] != spec:
            raise ValueError("Publication differs from its fixed task/model snapshot")
        if not existing["accepted"]:
            return current_models, existing
    try:
        report = train(
            history,
            root,
            config=config,
            initial_models=current_models,
            new_source_ids=new_source_ids,
            parent_version=parent,
        )
        candidate = tuple(
            load_model(root / name)[0] for name in ("target_kernel.pt", "historical_encoder.pt")
        )
    except (FloatingPointError, torch.linalg.LinAlgError) as error:
        result = {
            "accepted": False,
            "specification": spec,
            "parent_version": parent,
            "reason": "numerical_failure",
            "error": str(error),
        }
        write_json(publication_path, result)
        return current_models, result
    result = {
        "accepted": True,
        "specification": spec,
        "parent_version": parent,
        "reason": "finite_final_fixed_step_model",
        "training": report,
        "model_version": model_version(candidate[0]) + ":" + model_version(candidate[1]),
    }
    write_json(publication_path, result)
    write_json(
        root.parent / "active.json",
        {"directory": root.name, "model_version": result["model_version"]},
    )
    return candidate, result
