from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from experiments import candidate_sensitivity as sensitivity
from experiments.initialization import initial_indices
from experiments.joint_validation import Config
from experiments.scenarios import candidate_pool
from experiments.storage import read_json, write_json


class Bank:
    def __init__(self, scenes):
        self.candidates = scenes
        self.manifest = {"engineering_fixture": True}

    def __call__(self, scene):
        return scene.scenario_id


def test_candidate_subset_exposes_only_exact_prefix_and_never_other_coordinates():
    scenes = candidate_pool("front_brake", 11, replicate=30)
    bank = Bank(scenes)
    subset = sensitivity.CandidateSubset(bank, scenes[:4])
    assert subset.candidates == candidate_pool("front_brake", 4, replicate=30)
    assert subset(scenes[0]) == scenes[0].scenario_id
    with pytest.raises(ValueError, match="outside"):
        subset(scenes[4])
    with pytest.raises(ValueError, match="exact nested"):
        sensitivity.CandidateSubset(bank, list(reversed(scenes[:4])))


def test_scale_runner_builds_one_largest_bank_and_reuses_all_seven_baselines(tmp_path, monkeypatch):
    manifest = tmp_path / "data/manifest.json"
    write_json(manifest, {"engineering_fixture": True})
    collected, compared, classifiers = [], [], []

    def classifier_builder(history, destination):
        classifiers.append(destination)
        return {"engineering_fixture": True}

    def bank_loader(root, sut, kind, config, *, replicate):
        collected.append((sut.sut_id, config.candidate_count, replicate))
        return Bank(candidate_pool(kind, config.candidate_count, replicate=replicate))

    def compare(root, **kwargs):
        assert torch.get_num_threads() == 1
        config = kwargs["config"]
        assert kwargs["methods"] == sensitivity.MAIN_METHODS
        assert kwargs["candidate_replicate"] == 30
        assert kwargs["initial_pool_size"] == 4
        history = [SimpleNamespace(scenarios=[SimpleNamespace(kind="front_brake")])]
        assert kwargs["classifier_builder"](history, None) == {"engineering_fixture": True}
        rows = []
        for sut in kwargs["systems"]:
            bank = kwargs["bank_loader"](None, sut, "front_brake", config, replicate=30)
            assert len(bank.candidates) == config.candidate_count
            compared.append((sut.sut_id, bank.candidates))
            for method in kwargs["methods"]:
                rows.append(
                    {
                        "sut_id": sut.sut_id,
                        "scenario": "front_brake",
                        "seed": 11,
                        "method": method,
                        "result_path": f"methods/{sut.sut_id}/{method}/result.json",
                    }
                )
        return rows

    monkeypatch.setattr(sensitivity, "target_bank", bank_loader)
    monkeypatch.setattr(sensitivity, "fixed_comparison", compare)
    config = Config(candidate_count=11, budget=3, initial_count=1, scenarios=("front_brake",))
    torch.set_num_threads(2)
    rows = sensitivity.run(
        tmp_path / "experiment",
        config=config,
        sizes=(4, 7, 11),
        systems=CURRENT_INDEPENDENT_SUTS[:2],
        dataset_manifest=manifest,
        classifier_builder=classifier_builder,
        classifier_root=tmp_path / "frozen_classifiers",
    )
    assert len(rows) == 3 * 2 * 8
    assert len(collected) == 2
    assert classifiers == [tmp_path / "frozen_classifiers/front_brake"]
    assert all(count == 11 and replicate == 30 for _, count, replicate in collected)
    for sut in CURRENT_INDEPENDENT_SUTS[:2]:
        pools = [scenes for name, scenes in compared if name == sut.sut_id]
        assert pools[0] == pools[-1][:4] and pools[1] == pools[-1][:7]
    result = read_json(tmp_path / "experiment/reports/summary.json")
    assert result["status"] == "complete"
    assert result["physical_truth_records"] == 22 and result["logical_queries"] == 144
    assert read_json(tmp_path / "experiment/reports/protocol.json")["cpu_torch_threads"] == 1
    assert read_json(tmp_path / "experiment/reports/protocol.json")["initial_pool_size"] == 4
    assert (
        read_json(tmp_path / "experiment/reports/protocol.json")["historical_classifier_root"]
        == "../frozen_classifiers"
    )
    with pytest.raises(ValueError, match="front braking"):
        sensitivity.run(tmp_path, config=replace(config, scenarios=("car_following",)), sizes=(4,))


def test_initialization_uses_shared_prefix_and_preserves_single_pool_selection():
    assert initial_indices(9, 3, 11) == np.random.default_rng(11).permutation(9)[:3].tolist()
    assert initial_indices(9, 3, 11, pool_size=4) == initial_indices(4, 3, 11)
    for pool_size in (2, 10):
        with pytest.raises(ValueError, match="shared candidate prefix"):
            initial_indices(9, 3, 11, pool_size=pool_size)


def test_scale_comparison_rejects_unbound_initialization_before_any_execution(tmp_path):
    from experiments.pilot import fixed_comparison

    with pytest.raises(ValueError, match="explicit shared initialization pool"):
        fixed_comparison(tmp_path, suite="sensitivity")
