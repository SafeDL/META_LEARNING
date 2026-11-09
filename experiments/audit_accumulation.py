"""Verify sequential source admission, model publication and paid-task replay without queries."""

import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.training import Config as TrainingConfig
from testing_methods.history_transfer.training import load_model, specification

from .history import calculation_view, completed_archives, load_training_archives, model_version
from .joint_validation import OUTPUT, REGIMES
from .prepare_joint import DATASET_ROOT
from .rss import measurement_contract
from .storage import read_json, write_json


def load_pair(directory):
    return tuple(
        load_model(directory / name)[0] for name in ("target_kernel.pt", "historical_encoder.pt")
    )


def verify_sources(saved, history, models, task):
    spec, result = saved["specification"], saved["result"]
    assert spec["task_snapshot"] == task
    assert spec["feedback_mode"] == "joint" and spec["prior_mode"] == "sources"
    assert spec["historical_rss"] and spec["update_source_weights"]
    versions = [model_version(model) for model in models]
    assert [spec["target_model_version"], spec["encoder_version"]] == versions
    fingerprints = archive_fingerprints(history)
    for model in models:
        construction = model.construction_specification
        assert construction["scenario"] == task["scenario"]
        assert construction["held_family"] is None
        assert all(
            fingerprints.get(name) == fingerprint
            for name, fingerprint in construction["history_fingerprints"].items()
        )
    views = [calculation_view(archive) for archive in history]
    assert spec["source_snapshot"] == {
        "format": "joint-source-blocks",
        "scenario": task["scenario"],
        "model_version": versions[1],
        "archives": fingerprints,
        "rss_contract": measurement_contract(),
        "historical_rss": True,
        "views": {a.source_id: a.provenance["calculation_indices"] for a in views},
        "record_ids": {a.source_id: a.provenance.get("record_ids", []) for a in history},
        "view_record_ids": {a.source_id: a.provenance.get("record_ids", []) for a in views},
        "candidates": spec["candidates"],
        "covariance_mode": "point_joint_blocks",
    }
    assert all(scene["kind"] == task["scenario"] for scene in spec["candidates"])
    assert result["component_names"] == [
        *[archive.source_id for archive in history],
        "independent_target",
        "physical_target",
    ]


def verify_replay(replay, history, completed, before, after, config):
    assert replay["new_physical_calls"] == replay["new_paid_queries"] == 0
    assert replay["publication_gate"] is False
    assert replay["specification"] == {
        "before": [model_version(model) for model in before],
        "after": [model_version(model) for model in after],
        "history": archive_fingerprints(history),
        "tasks": archive_fingerprints(completed),
        **{name: config[name] for name in ("initial_count", "batch_size", "device")},
    }
    rows = {row["source_id"]: row for row in replay["tasks"]}
    assert len(rows) == len(replay["tasks"]) == len(completed)
    assert set(rows) == {archive.source_id for archive in completed}
    for archive in completed:
        row = rows[archive.source_id]
        assert row["record_ids"] == archive.provenance["record_ids"]
        start = min(config["initial_count"], max(0, len(archive.scenarios) - 1))
        for side, models in (("before", before), ("after", after)):
            entry = row[side]
            probability = np.asarray(entry["prequery_probabilities"])
            assert probability.shape == archive.collision.shape
            assert (
                np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            )
            assert entry["initial_count"] == start
            assert entry["evaluated_records"] == len(archive.scenarios) - start
            p = np.clip(probability[start:], 1e-12, 1 - 1e-12)
            y = archive.collision[start:].astype(float)
            np.testing.assert_allclose(
                [entry["collision_nll"], entry["brier"]],
                [-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)), np.mean((p - y) ** 2)],
                rtol=1e-12,
                atol=1e-12,
            )
            assert entry["shared_training_contains_task"] == any(
                archive.source_id in model.construction_specification["history_fingerprints"]
                for model in models
            )
        np.testing.assert_allclose(
            [row["nll_change"], row["brier_change"]],
            [row["after"][key] - row["before"][key] for key in ("collision_nll", "brier")],
            rtol=1e-12,
            atol=1e-12,
        )


def read_completed(path, task, sut, saved):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    assert archive["completed"] and archive["completed_order"] == task["ordinal"]
    assert archive["branch_id"] == task["branch_id"]
    assert archive["sut_id"] == sut["sut_id"] and archive["sut_version"] == sut["version"]
    assert archive["original_role"] == sut["role"] != "development"
    assert archive["model_snapshot"] == {
        name: saved["specification"][name] for name in ("target_model_version", "encoder_version")
    }
    assert archive["attempts"] == saved["query_ledger"]
    assert len(archive["executions"]) == saved["specification"]["budget"]
    completed = completed_archives(
        [path], before_order=task["ordinal"] + 1, branch_id=task["branch_id"]
    )
    assert len(completed) == 1
    response = completed[0]
    assert response.role == "target" and response.provenance["original_role"] == sut["role"]
    assert all(scene.kind == task["scenario"] for scene in response.scenarios)
    assert response.provenance["record_ids"] == [e["record_id"] for e in archive["executions"]]
    return response


def audit(
    root=OUTPUT / "sequential",
    *,
    dataset_manifest=DATASET_ROOT / "manifest.json",
    model_root=OUTPUT / "models",
):
    root, model_root = Path(root), Path(model_root)
    protocol, summary = (
        read_json(root / "reports" / name) for name in ("protocol.json", "summary.json")
    )
    assert protocol["suite"] == "sequential"
    config = protocol["config"]
    kinds, systems = config["scenarios"], protocol["systems"]
    history = {kind: load_training_archives(dataset_manifest, kind) for kind in kinds}
    initial = {kind: load_pair(model_root / kind / "all_sources") for kind in kinds}
    for kind, models in initial.items():
        for model in models:
            assert model.construction_specification["history_fingerprints"] == archive_fingerprints(
                history[kind]
            )
    rows = {(r["seed"], r["regime"], r["ordinal"], r["scenario"]): r for r in summary["runs"]}
    updates = {(u["branch_id"], u["after_systems"], u["scenario"]): u for u in summary["updates"]}
    assert len(rows) == len(summary["runs"]) and len(updates) == len(summary["updates"])
    checked, publications, visited, visited_updates = [], [], set(), set()
    zero_references = {}
    for seed in config["algorithm_seeds"]:
        order = np.random.default_rng(seed).permutation(len(systems))
        for regime in REGIMES:
            branch_id = f"{regime}:{seed}"
            branch = root / "methods" / regime / str(seed)
            active, past, learned = dict(initial), {k: [] for k in kinds}, {k: set() for k in kinds}
            for ordinal, index in enumerate(order, 1):
                sut = systems[index]
                keys = [(seed, regime, ordinal, kind) for kind in kinds]
                if not all(key in rows for key in keys):
                    break
                completed = {}
                for key in keys:
                    row, kind = rows[key], key[-1]
                    task = {
                        **{name: sut[name] for name in ("sut_id", "family", "lineage")},
                        "scenario": kind,
                        "ordinal": ordinal,
                        "regime": regime,
                        "seed": seed,
                        "branch_id": branch_id,
                    }
                    assert all(row[name] == value for name, value in task.items())
                    admitted = (
                        history[kind]
                        if regime == "fixed_history"
                        else [*history[kind], *past[kind]]
                    )
                    assert row["source_ids"] == [a.source_id for a in admitted]
                    assert not any(a.provenance.get("sut_id") == sut["sut_id"] for a in past[kind])
                    saved = read_json(root / row["result_path"])
                    verify_sources(saved, admitted, active[kind], task)
                    zero = saved["result"]["prediction_snapshots"][0]
                    assert zero["paid_queries"] == 0 and zero["observed_indices"] == []
                    if ordinal == 1:
                        reference = zero_references.setdefault((seed, kind), zero["probabilities"])
                        np.testing.assert_allclose(
                            zero["probabilities"], reference, rtol=1e-12, atol=1e-12
                        )
                    path = branch / "completed" / f"{ordinal:02d}_{sut['sut_id']}_{kind}.json.gz"
                    completed[kind] = read_completed(path, task, sut, saved)
                    checked.append(
                        {
                            **task,
                            "sources": row["source_ids"],
                            "model_versions": [model_version(m) for m in active[kind]],
                        }
                    )
                    visited.add(key)
                # Admit this ADS only after every registered functional scenario has completed.
                for kind in kinds:
                    past[kind].append(completed[kind])
                if (
                    regime != "update_shared"
                    or ordinal % config["update_every_systems"]
                    or ordinal == len(order)
                ):
                    continue
                for kind in kinds:
                    key = branch_id, ordinal, kind
                    update = updates[key]
                    folder = branch / "models" / kind / f"after_{ordinal:02d}_systems"
                    publication = read_json(folder / "publication.json")
                    assert update["publication"] == publication
                    admitted = [*history[kind], *past[kind]]
                    new_ids = [a.source_id for a in past[kind] if a.source_id not in learned[kind]]
                    parent = ":".join(model_version(model) for model in active[kind])
                    expected = specification(
                        admitted, None, TrainingConfig(**protocol["shared_update_training"])
                    )
                    expected.update(new_source_ids=new_ids, parent_version=parent)
                    assert publication["specification"] == expected
                    assert publication["parent_version"] == parent
                    if publication["accepted"]:
                        assert publication["reason"] == "finite_final_fixed_step_model"
                        for part, filename in (
                            ("target", "target_kernel.json"),
                            ("encoder", "historical_encoder.json"),
                        ):
                            training = read_json(folder / filename)
                            assert publication["training"][part] == training
                            assert training["status"] == "complete"
                            assert training["specification"] == expected
                            assert training["final_step"] == expected["config"]["steps"]
                            assert training["curve"][-1]["step"] == training["final_step"]
                            assert (
                                sum(p["diagnostic_steps"] for p in training["curve"])
                                == training["final_step"]
                            )
                        after = load_pair(folder)
                        assert all(model.construction_specification == expected for model in after)
                        assert publication["model_version"] == ":".join(
                            model_version(model) for model in after
                        )
                        assert (root / update["retention_path"]).resolve() == (
                            folder / "retention.json"
                        ).resolve()
                        verify_replay(
                            read_json(folder / "retention.json"),
                            admitted,
                            past[kind],
                            active[kind],
                            after,
                            config,
                        )
                        active[kind] = after
                        learned[kind].update(new_ids)
                    else:
                        assert publication["reason"] == "numerical_failure" and publication["error"]
                        assert update["retention_path"] is None
                    publications.append(
                        {
                            "branch_id": branch_id,
                            "after_systems": ordinal,
                            "scenario": kind,
                            "accepted": publication["accepted"],
                            "trained_sources": list(expected["history_fingerprints"]),
                            "new_sources": new_ids,
                        }
                    )
                    visited_updates.add(key)
    assert visited == set(rows), "Saved tasks are not complete chronological ADS groups"
    assert visited_updates == set(updates), (
        "Publication does not follow a completed update boundary"
    )
    expected_runs = len(systems) * len(kinds) * len(REGIMES) * len(config["algorithm_seeds"])
    if summary["status"] == "complete":
        assert len(checked) == expected_runs
    result = {
        "format": "sequential-source-and-publication-check",
        "status": "complete" if len(checked) == expected_runs else "partial",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "expected_runs": expected_runs,
        "checked_runs": len(checked),
        "checked_publications": len(publications),
        "runs": checked,
        "publications": publications,
        "verification_scope": "Branch/order admission, original paid-record identity, source/cache snapshots, actual model versions and publication training scope; replay scores recalculated from saved probabilities and original paid labels. Prediction inference is not rerun.",
        "new_queries_or_simulations": 0,
    }
    write_json(root / "reports/source_admission_check.json", result)
    return result


if __name__ == "__main__":
    report = audit()
    print({key: value for key, value in report.items() if key not in ("runs", "publications")})
