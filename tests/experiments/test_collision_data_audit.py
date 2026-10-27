import numpy as np
import pytest

from experiments.audit_collision_data import verify_archive
from experiments.executions import collect_bank
from experiments.scenarios import sample
from experiments.storage import load_archive, save_archive
from testing_methods.common.response import ResponseArchive


def write_archive(root, bank):
    executions = [bank(s) for s in bank.candidates]
    provenance = {
        "qualified": True,
        "bank": str(bank.root.relative_to(root)),
        "contract": bank.manifest["contract"],
        "environment_seed": 81001,
        "physical_calls": bank.manifest["attempted_calls"],
    }
    archive = ResponseArchive(
        "history_idm_mobil",
        "rule",
        "idm_mobil",
        "history",
        bank.candidates,
        np.array([e.feedback.collision for e in executions]),
        np.array(
            [[e.feedback.safety.min_clearance_m, e.feedback.safety.min_ttc_s] for e in executions]
        ),
        provenance,
    )
    path = root / "history/front_brake/history_idm_mobil.npz"
    save_archive(path, archive, provenance=provenance)
    return path


def test_audit_rejects_validly_encoded_collision_labels_that_contradict_real_trajectories(tmp_path):
    scenes = sample("front_brake", 2, "history")
    bank = collect_bank(
        "history_idm_mobil",
        scenes,
        tmp_path / "physical/history/front_brake/history_idm_mobil",
        workers=1,
    )
    path = write_archive(tmp_path, bank)
    assert verify_archive(path, records_root=tmp_path, candidate_count=2)["responses"] == 2
    wrong_identity = path.with_name("history_fvdm_lane.npz")
    wrong_identity.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="canonical archive path"):
        verify_archive(wrong_identity, records_root=tmp_path, candidate_count=2)
    changed = load_archive(path)
    changed.collision[0] = not changed.collision[0]
    path.unlink()
    save_archive(path, changed, provenance=changed.provenance)
    with pytest.raises(ValueError, match="collision labels disagree"):
        verify_archive(path, records_root=tmp_path, candidate_count=2)


def test_audit_accounts_for_a_completed_call_whose_response_was_lost_before_resume(tmp_path):
    scenes = sample("front_brake", 2, "history")
    destination = tmp_path / "physical/history/front_brake/history_idm_mobil"
    collect_bank("history_idm_mobil", scenes, destination, workers=1)
    # Simulate interruption after execution/attempt logging but before retaining this artifact.
    (destination / "executions/000000.json.gz").unlink()
    bank = collect_bank("history_idm_mobil", scenes, destination, workers=1)
    report = verify_archive(write_archive(tmp_path, bank), records_root=tmp_path, candidate_count=2)
    assert report["responses"] == 2
    assert report["actual_physical_calls"] == report["successful_physical_calls"] == 3
    assert report["successful_calls_without_retained_response"] == 1
    assert report["physical_execution_s"] > report["retained_response_execution_s"]
