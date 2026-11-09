import copy
import gzip
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from controllers.highway.control import Control
from experiments import joint_validation as workflow
from experiments.audit_accumulation import audit, load_pair
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from experiments.storage import read_json, save_archive, write_json
from simulator.highway.collision_env import execute
from testing_methods.common.response import ResponseArchive, archive_fingerprints
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.training import Config, save_model, specification


class Brake:
    def act(self, world):
        return Control(-8, 0)


class Bank:
    def __init__(self, scenes, root):
        self.root = root
        self.candidates = scenes
        self.manifest = {"contract": {"fixture": "audit test"}, "environment_seed": 81001}
        self.records = {s.scenario_id: execute(Brake(), s, duration_s=0.1) for s in scenes}

    def __call__(self, scene):
        return self.records[scene.scenario_id]

    def execution_path(self, index):
        path = self.root / f"{index:06d}.json.gz"
        scene = self.candidates[index]
        actual = self(scene)
        write_json(path, {"scenario": asdict(scene), "trajectory": actual.trajectory})
        return path


@pytest.fixture(scope="module")
def completed_sequence(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("accumulation_audit")
    kind = "car_following"
    history, entries = [], []
    for i in range(2):
        scenes = sample(kind, 4, "history", replicate=i)
        archive = ResponseArchive(
            f"history_{i}",
            "rule",
            "idm",
            "history",
            scenes,
            np.array([False, True, False, True]),
            np.ones((4, 2)),
            provenance={"record_ids": [f"{i}:{n}" for n in range(4)]},
            rss_margin=np.array([0.5, -0.5, 0.2, -0.2]),
            rss_contract=measurement_contract(),
        )
        path = temporary / f"history_{i}.npz"
        save_archive(path, archive, provenance=archive.provenance)
        history.append(archive)
        entries.append(
            {
                "source_id": archive.source_id,
                "scenario": kind,
                "path": path.name,
                "fingerprint": archive_fingerprints([archive])[archive.source_id],
            }
        )
    manifest = temporary / "manifest.json"
    write_json(manifest, {"archives": entries})
    training = Config(
        steps=1, initial_count=1, budget=3, historical_context=2, diagnostic_interval=1
    )
    spec = specification(history, None, training)
    model_root = temporary / "models"
    folder = model_root / kind / "all_sources"
    folder.mkdir(parents=True)
    torch.manual_seed(7301)
    for name in ("target_kernel", "historical_encoder"):
        save_model(folder / f"{name}.pt", JointResponseKernel(3).double(), name, spec, 1)
    banks = {}

    def factory(root, sut, kind, config, **kwargs):
        if sut.sut_id not in banks:
            banks[sut.sut_id] = Bank(
                sample(kind, config.candidate_count, "new", replicate=20), root / sut.sut_id / kind
            )
        return banks[sut.sut_id]

    root = temporary / "sequential"
    config = workflow.Config(
        candidate_count=4,
        budget=2,
        initial_count=1,
        scenarios=(kind,),
        algorithm_seeds=(11,),
        batch_size=2,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(workflow, "target_bank", factory)
        workflow.sequential(
            root,
            config=config,
            systems=CURRENT_INDEPENDENT_SUTS[:3],
            dataset_manifest=manifest,
            model_loader=lambda k: load_pair(folder),
            training_config=training,
        )
    return root, {"dataset_manifest": manifest, "model_root": model_root}


def test_full_sequence_and_shared_model_replay_are_independently_checked(completed_sequence):
    root, options = completed_sequence
    result = audit(root, **options)
    assert result["status"] == "complete" and result["checked_runs"] == 9
    assert result["checked_publications"] == 1
    assert result["new_queries_or_simulations"] == 0
    for regime in workflow.REGIMES:
        counts = [len(r["sources"]) for r in result["runs"] if r["regime"] == regime]
        assert counts == ([2, 2, 2] if regime == "fixed_history" else [2, 3, 4])


@pytest.mark.parametrize(
    "corruption",
    [
        "foreign_source",
        "wrong_model",
        "wrong_branch",
        "training_scope",
        "training_steps",
        "replay_score",
        "replay_membership",
        "model_weights",
        "missing_middle_task",
    ],
)
def test_saved_scope_and_publication_errors_are_rejected(completed_sequence, corruption):
    root, options = completed_sequence
    summary = read_json(root / "reports/summary.json")
    row = next(r for r in summary["runs"] if r["regime"] == "update_shared" and r["ordinal"] == 3)
    publication = (
        root / "methods/update_shared/11/models/car_following/after_02_systems/publication.json"
    )
    if corruption in ("foreign_source", "wrong_model"):
        path = root / row["result_path"]
    elif corruption == "wrong_branch":
        path = (
            root
            / "methods/update_shared/11/completed"
            / f"03_{row['sut_id']}_car_following.json.gz"
        )
    elif corruption in ("training_scope", "training_steps"):
        path = publication
    elif corruption in ("replay_score", "replay_membership"):
        path = publication.parent / "retention.json"
    elif corruption == "model_weights":
        path = publication.parent / "target_kernel.pt"
    else:
        path = root / "reports/summary.json"
    original = path.read_bytes()
    summary_path = root / "reports/summary.json"
    summary_original = summary_path.read_bytes()
    training_path = publication.parent / "target_kernel.json"
    training_original = training_path.read_bytes()
    try:
        if corruption == "model_weights":
            value = torch.load(path, weights_only=True)
            value["state_dict"][next(iter(value["state_dict"]))] += 0.001
            torch.save(value, path)
        else:
            value = json.loads(gzip.decompress(original) if path.suffix == ".gz" else original)
            if corruption == "foreign_source":
                value["specification"]["source_snapshot"]["archives"]["future_target"] = "unpaid"
            elif corruption == "wrong_model":
                value["specification"]["encoder_version"] = "model_changed_mid_task"
            elif corruption == "wrong_branch":
                value["branch_id"] = "append_sources:11"
            elif corruption == "training_scope":
                value["specification"]["new_source_ids"] = []
                summary["updates"][0]["publication"] = copy.deepcopy(value)
                write_json(summary_path, summary)
            elif corruption == "training_steps":
                value["training"]["target"]["final_step"] = 0
                write_json(training_path, value["training"]["target"])
                summary["updates"][0]["publication"] = copy.deepcopy(value)
                write_json(summary_path, summary)
            elif corruption == "replay_score":
                value["tasks"][0]["after"]["collision_nll"] += 0.1
            elif corruption == "replay_membership":
                value["tasks"][0]["after"]["shared_training_contains_task"] = False
            else:
                value["runs"] = [
                    r
                    for r in value["runs"]
                    if not (r["regime"] == "fixed_history" and r["ordinal"] == 2)
                ]
                value["status"] = "running"
            if path.suffix == ".gz":
                path.write_bytes(gzip.compress(json.dumps(value).encode()))
            else:
                write_json(path, value)
        with pytest.raises((AssertionError, ValueError)):
            audit(root, **options)
    finally:
        path.write_bytes(original)
        summary_path.write_bytes(summary_original)
        training_path.write_bytes(training_original)


def test_partial_completed_ads_prefix_is_reported_as_partial(completed_sequence):
    root, options = completed_sequence
    path = root / "reports/summary.json"
    original = path.read_bytes()
    summary = json.loads(original)
    summary["runs"] = [
        r for r in summary["runs"] if r["regime"] == "fixed_history" and r["ordinal"] == 1
    ]
    summary.update(status="running", updates=[])
    try:
        write_json(path, summary)
        report = audit(root, **options)
        assert report["status"] == "partial" and report["checked_runs"] == 1
    finally:
        path.write_bytes(original)
