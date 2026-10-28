"""Match recorded driving-system provenance across the reviewed directory reorganization."""

import json

from .protocol import ROOT

REGISTRATION = (
    ROOT
    / "experiment_outputs/benchmark/collision/discovery/independent_validation/model_construction"
)


def maintained_path(name):
    for old, new in (
        ("benchmark/", "experiments/"),
        ("simulation/", "simulator/"),
        ("methods/gcrt/", "testing_methods/history_transfer/"),
        ("methods/baselines/fsbo/", "testing_methods/common/fsbo/"),
        ("methods/", "testing_methods/"),
        ("results/", "experiment_outputs/"),
    ):
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def source_contract_matches(recorded, current):
    """Accept unchanged source or the explicitly reviewed directory reorganization."""
    if recorded == current:
        return True
    maintenance = json.loads((REGISTRATION / "source_maintenance.json").read_text(encoding="utf-8"))
    reviewed = maintenance["source_reorganization"]["relocated_source_files"]
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
