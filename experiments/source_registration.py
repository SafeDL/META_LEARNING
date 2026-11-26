"""Match recorded driving-system provenance across the reviewed directory reorganization."""

import json

from .protocol import RESULTS_ROOT

SOURCE_REORGANIZATION = RESULTS_ROOT / "provenance/source_reorganization.json"
ARTIFACT_MAINTENANCE = RESULTS_ROOT / "provenance/reorganization.json"
CODE_MAINTENANCE = RESULTS_ROOT / "provenance/source_changes.json"

# These paths occur in frozen manifests and source contracts. New writers use
# the current layout directly; the recorded bytes remain unchanged.
ARTIFACT_RELOCATIONS = (
    ("test_records", "results/data"),
    (
        "experiment_outputs/benchmark/collision/qualification",
        "results/k_shot_adaptation/qualification",
    ),
    (
        "artifacts/runs/qualification/reports",
        "results/k_shot_adaptation/qualification",
    ),
    (
        "experiment_outputs/benchmark/collision/interface.json",
        "results/provenance/interface.json",
    ),
    (
        "artifacts/runs/preparation/reports/interface.json",
        "results/provenance/interface.json",
    ),
    (
        "experiment_outputs/gcrt/collision/joint_response_models",
        "results/joint_response/models",
    ),
    ("experiment_outputs/gcrt/collision/joint_response", "results/joint_response"),
    ("experiment_outputs/metadrive", "results/metadrive"),
    ("artifacts/runs/highway_learning_ads", "results/public_ads/qualification"),
    ("artifacts/runs/braking_scene_design", "results/scene_design/braking"),
    ("artifacts/runs/interaction_scene_design", "results/scene_design/interaction"),
    ("artifacts/runs/safe_scene_replay", "results/scene_replays/safe"),
    ("artifacts/runs/collision_scene_replay", "results/scene_replays/collision"),
    ("artifacts/runs", "results"),
)


def maintained_path(name):
    name = str(name).replace("\\", "/")
    if name.startswith(("results/benchmark/", "results/gcrt/", "results/metadrive/")):
        name = "experiment_outputs/" + name[len("results/") :]
    for old, new in ARTIFACT_RELOCATIONS:
        if name == old or name.startswith(old + "/"):
            return new + name[len(old) :]
    for old, new in (
        ("benchmark/", "experiments/"),
        ("simulation/", "simulator/"),
        ("methods/gcrt/", "testing_methods/history_transfer/"),
        ("methods/", "testing_methods/"),
    ):
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def source_contract_matches(recorded, current):
    """Accept identical or explicitly reviewed source identities."""
    if recorded == current:
        return True
    changes = json.loads(CODE_MAINTENANCE.read_text(encoding="utf-8"))["source_changes"]
    if recorded.keys() == current.keys() and all(
        recorded[name] == digest
        or (
            name in changes
            and digest == changes[name]["after_sha256"]
            and recorded[name]
            in [changes[name]["before_sha256"], *changes[name].get("reviewed_sha256", [])]
        )
        for name, digest in current.items()
    ):
        return True
    for path in (CODE_MAINTENANCE, ARTIFACT_MAINTENANCE):
        changes = json.loads(path.read_text(encoding="utf-8"))["source_changes"]
        current = {
            name: changes[name]["before_sha256"]
            if name in changes and digest == changes[name]["after_sha256"]
            else digest
            for name, digest in current.items()
        }
        if recorded == current:
            return True
    reviewed = json.loads(SOURCE_REORGANIZATION.read_text(encoding="utf-8"))[
        "relocated_source_files"
    ]
    if {maintained_path(name) for name in recorded} != set(current):
        return False
    return all(
        current.get(name) == digest
        or (
            name in reviewed
            and reviewed[name]["before_sha256"] == digest
            and reviewed[name]["after_sha256"] == current[maintained_path(name)]
        )
        for name, digest in recorded.items()
    )


def specification_matches(recorded, current):
    if recorded.keys() != current.keys():
        return False
    return all(
        source_contract_matches(recorded[name], value)
        if name
        in ("training_contract", "source_files", "policy_training_source", "source_contract")
        else recorded[name] == value
        for name, value in current.items()
    )
