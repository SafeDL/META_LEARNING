import numpy as np
import pytest

from experiments.scenarios import features, sample
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.joint_training import TRAINING, load_model, train


def archives(role):
    scenarios = sample("cut_in", 24, role)
    x = features(scenarios)
    return [
        ResponseArchive(
            f"{role}_{name}",
            family,
            name,
            role,
            scenarios,
            x[:, 0] > threshold,
            np.column_stack((0.1 + 3 * x[:, 1], 1 + 2 * x[:, 2])),
        )
        for name, family, threshold in (("a", "rule", 0.6), ("b", "search", 0.7))
    ]


def test_joint_training_builds_and_reuses_only_complete_role_checked_models(tmp_path):
    history, development = archives("history"), archives("development")
    config = {
        **TRAINING,
        "steps": 2,
        "validation_interval": 1,
        "support_counts": (2, 4),
        "query_count": 2,
        "adaptation_steps": 1,
    }
    path = tmp_path / "model.pt"
    report = train(history, development, path, uncorrelated=True, config=config)
    assert report["status"] == "complete"
    assert len(report["curve"]) == 2
    assert all(len(row["episodes"]) == 4 for row in report["curve"])
    model, metadata = load_model(path)
    assert model.correlation == 0
    assert model.raw_correlation.requires_grad is False
    assert metadata["specification"]["uncorrelated"] is True
    assert train(history, development, path, uncorrelated=True, config=config) == report
    with pytest.raises(ValueError, match="different inputs"):
        train(history, development, path, uncorrelated=True, config={**config, "seed": 9})


def test_joint_training_rejects_role_and_family_leakage_before_writing(tmp_path):
    history, development = archives("history"), archives("development")
    with pytest.raises(ValueError, match="disjoint"):
        train(history, history, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Held family"):
        train(history, development, tmp_path / "held.pt", held_family="rule")
    assert list(tmp_path.iterdir()) == []
