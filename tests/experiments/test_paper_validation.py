from dataclasses import asdict

import numpy as np
import pytest
import torch

from experiments.executions import ExecutionBank
from experiments.paper_validation import freeze
from experiments.scenarios import sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.storage import write_json
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_composite_bank_routes_each_coordinate_to_its_original_physical_record(tmp_path):
    scenes = sample("car_following", 2, "history", replicate=81)
    paths = [tmp_path / "old/record.json.gz", tmp_path / "supplement/record.json.gz"]
    for index, (scene, path) in enumerate(zip(scenes, paths)):
        feedback = QueryFeedback(scene.scenario_id, bool(index), SafetyMeasurements(1, 2), 0.1)
        write_json(
            path,
            {
                "scenario": asdict(scene),
                "feedback": feedback.to_dict(),
                "trajectory": [{"time_s": 0, "collision": bool(index)}],
            },
        )
    write_json(
        tmp_path / "merged/manifest.json",
        {
            "format": "collision-execution-bank",
            "status": "complete",
            "scenarios": [asdict(s) for s in scenes],
            "execution_refs": [str(p) for p in paths],
        },
    )
    bank = ExecutionBank(tmp_path / "merged")
    assert not (bank.root / "executions").exists()
    assert bank.execution_path(0) == paths[0] and bank.execution_path(1) == paths[1]
    assert bank(scenes[0]).feedback.collision is False
    assert bank(scenes[1]).feedback.collision is True
    write_json(paths[0], {"scenario": asdict(scenes[1])})
    with pytest.raises(ValueError, match="different coordinate"):
        bank(scenes[0])


def test_registration_rejects_candidate_or_model_changes_instead_of_overwriting(tmp_path):
    path = tmp_path / "registration.json"
    freeze(path, {"model": "a", "candidates": [1, 2]})
    freeze(path, {"model": "a", "candidates": [1, 2]})
    with pytest.raises(ValueError, match="inputs changed"):
        freeze(path, {"model": "b", "candidates": [1, 2]})


@pytest.mark.parametrize("kernel", (JointResponseKernel, SharedPrivateKernel))
def test_same_paid_prefix_gives_identical_raw_component_and_no_history_predictions(kernel):
    from experiments.rss import measurement_contract

    torch.set_num_threads(1)
    scenes = sample("car_following", 8, "history", replicate=83)

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                index == 1,
                SafetyMeasurements(1, 1, rss_margin=-0.5 if index == 1 else 0.3),
                0,
                measurement_contract(),
            )

    options = dict(
        candidates=scenes,
        oracle=Oracle(),
        initial_indices=[0, 1],
        budget=3,
        query_order=[0, 1, 2],
        snapshot_counts=(2,),
        batch_size=3,
    )
    full = select(pretrained=kernel(3).double(), **options)
    raw = select(raw_kernel_type=kernel, **options)
    np.testing.assert_allclose(
        full["queries"][2]["prequery_component_probabilities"][-1],
        raw["queries"][2]["prequery_probability"],
        atol=1e-8,
        rtol=0,
    )
