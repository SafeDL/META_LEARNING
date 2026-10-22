import json
from dataclasses import replace

import numpy as np
import pytest

from benchmark.collision_runs import run_task
from benchmark.executions import collect_bank
from benchmark.scenarios import sample
from methods.common.response import ResponseArchive


def test_end_to_end_paid_replay_keeps_identity_outside_selector_and_reconstructs_metrics(tmp_path):
    scenes = sample("front_brake", 6, "development", replicate=101)
    bank = collect_bank("history_idm_mobil", scenes, tmp_path / "bank", workers=2)
    destination = tmp_path / "run.json"
    metadata = {"target_id": "history_idm_mobil", "lineage": "idm_mobil",
                "scenario": "front_brake", "condition": "development", "environment_seed": 81001}
    output = run_task(bank, (), method="random", algorithm_seed=11, budget=4,
                      initial_count=2, destination=destination, metadata=metadata)
    assert output["metrics"]["queries"] == 4
    assert output["cost"]["actual_new_physical_calls"] == 0
    assert output["cost"]["physical_bank_calls"] == 6
    assert len(output["query_ledger"]) == 4
    assert "target_id" not in output["result"]
    assert "history_idm_mobil" not in json.dumps(output["result"])
    assert run_task(bank, (), method="random", algorithm_seed=11, budget=4,
                    initial_count=2, destination=destination, metadata=metadata) == output


@pytest.mark.parametrize("method", ["history_response", "fsbo", "pooled_ft"])
def test_learned_models_cross_the_isolated_boundary_without_target_identity(tmp_path, method):
    from methods.gcrt.config import ModelConfig
    from methods.gcrt.train import train as train_main
    from methods.baselines.fsbo.method import CONFIG as fsbo_config, train as train_fsbo
    from methods.baselines.pooled_ft.method import CONFIG as pooled_config, train as train_pooled

    history = []
    for index in range(2):
        scenes = sample("front_brake", 48, "history", replicate=index)
        history.append(ResponseArchive(f"fixture_{index}", "rule", f"fixture_{index}", "history",
                                       scenes, np.arange(48) % 3 == 0, np.ones((48, 2))))
    development = [ResponseArchive("fixture_dev", "rule", "fixture_dev", "development",
                                    sample("front_brake", 48, "development"),
                                    np.arange(48) % 3 == 0, np.ones((48, 2)))]
    checkpoint = tmp_path / "private_target_family_name.pt"
    if method == "history_response":
        train_main(history, development, checkpoint, config=replace(
            ModelConfig(), steps=2, validation_interval=1, support_counts=(4,), query_count=8))
    else:
        configuration = dict(fsbo_config if method == "fsbo" else pooled_config)
        configuration.update(steps=2, validation_interval=1, support_count=4,
                             query_count=8, batch_size=12, adaptation_steps=1)
        (train_fsbo if method == "fsbo" else train_pooled)(history, development, checkpoint,
                                                          config=configuration)
    scenes = sample("front_brake", 6, "development", replicate=102)
    bank = collect_bank("history_idm_mobil", scenes, tmp_path / "bank", workers=2)
    result = run_task(bank, history, method=method, algorithm_seed=11, budget=4,
                      initial_count=2, destination=tmp_path / "run.json",
                      metadata={"target_id": "private_target_name"}, checkpoint=checkpoint)
    assert result["result"]["actual_queries"] == 4
    serialized = json.dumps(result["result"])
    assert "private_target" not in serialized
