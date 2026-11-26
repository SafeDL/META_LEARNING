import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments import baselines
from experiments.rss import measurement_contract
from experiments.scenarios import candidate_pool
from testing_methods.common.response import ResponseArchive


def histories(kind, count=12):
    scenes = candidate_pool(kind, count, replicate=98)
    return [
        ResponseArchive(
            f"history_{i}",
            "rule",
            f"route_{i}",
            "history",
            scenes,
            np.arange(count) % 3 == i,
            np.ones((count, 2)),
            rss_margin=np.linspace(-0.8, 0.4, count),
            rss_contract=measurement_contract(),
        )
        for i in range(2)
    ]


def test_history_classifiers_load_existing_models_without_retraining_or_rewriting(
    tmp_path, monkeypatch
):
    import torch

    names = ("scenario_fuzz",)
    monkeypatch.setattr(
        baselines.fuzz_model,
        "DEFAULT",
        replace(baselines.fuzz_model.DEFAULT, epochs=1, hidden_dim=4, heads=1),
    )
    history = histories("car_following", 8)
    original = baselines.historical_classifiers(history, tmp_path, names=names)
    assert set(original) == set(names)
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == set(names)
    paths = list(tmp_path.rglob("*.*"))
    times = {path: path.stat().st_mtime_ns for path in paths}

    def reject_training(*args, **kwargs):
        pytest.fail("Shared classifiers must be loaded without new training")

    monkeypatch.setattr(baselines.fuzz_model, "train", reject_training)
    loaded = baselines.historical_classifiers(history, tmp_path, train_missing=False, names=names)
    for name in original:
        for key, tensor in original[name].state_dict().items():
            assert torch.equal(tensor, loaded[name].state_dict()[key])
    assert times == {path: path.stat().st_mtime_ns for path in paths}
    with pytest.raises(FileNotFoundError):
        baselines.historical_classifiers(
            history, tmp_path / "missing", train_missing=False, names=names
        )
    changed = copy.deepcopy(history)
    changed[0].rss_margin[0] += 0.1
    with pytest.raises(ValueError, match="historical construction"):
        baselines.historical_classifiers(changed, tmp_path, train_missing=False, names=names)
