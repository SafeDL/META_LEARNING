from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.response_baselines.pooled_gp import predict_pooled, select
from testing_methods.common.classification import torch_moments
from testing_methods.history_transfer.inference import JointPosterior, adapted_model
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from tests.fixtures.feedback import Oracle


def test_pooled_matches_direct_raw_joint_posterior_with_conflicting_duplicate_coordinates():
    torch.set_num_threads(1)
    base = SharedPrivateKernel(3, deep=False).double()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    x = np.array([[0.1, 0.3, 0.5], [0.1, 0.3, 0.5], [0.8, 0.5, 0.4]])
    labels, margins = [False, True, False], [-0.2, 0.3, 0.6]
    query = np.array([[0.2, 0.4, 0.5], [0.9, 0.1, 0.2]])
    original = {name: value.clone() for name, value in base.state_dict().items()}
    probability, _, _ = predict_pooled(base, x, labels, margins, query, batch_size=1)
    model = adapted_model(base, x, labels, margins)
    with torch.no_grad():
        state = JointPosterior(model.point_mean(3), model.kernel(x), labels, margins)
        mean, blocks = state.predict(
            model.point_mean(2), model.kernel(query, x), model.point_covariance(2)
        )
        expected = torch_moments(mean[:, 0], blocks[:, 0, 0])[0]
    np.testing.assert_allclose(probability, expected, atol=1e-12, rtol=1e-12)
    assert state.count == 3
    assert all(torch.equal(original[name], value) for name, value in base.state_dict().items())


@pytest.mark.parametrize("inference_backend", ["dense", "grouped"])
def test_pooled_prefix_checkpoints_are_causal_and_restart_from_the_frozen_base(
    tmp_path, inference_backend
):
    scenes = sample("car_following", 8, "new", replicate=90)
    x = features(scenes)
    base = SharedPrivateKernel(3, deep=False).double()
    history = {"x": x[:2], "collision": [False, True], "rss_margin": [-0.4, 0.2]}
    labels, margins = np.arange(8) % 2 == 0, np.linspace(-0.7, 0.6, 8)
    order = [3, 1, 6]
    oracle = Oracle(scenes, labels, margins)
    full = select(
        scenes,
        oracle,
        pretrained=base,
        history=history,
        query_order=order,
        budget=3,
        snapshot_counts=(0, 1, 3),
        checkpoint_path=tmp_path / "partial.json",
        inference_backend=inference_backend,
    )
    changed_labels, changed_margins = labels.copy(), margins.copy()
    changed_labels[[1, 6, 7]] = ~changed_labels[[1, 6, 7]]
    changed_margins[[1, 6, 7]] += 0.5
    short = select(
        scenes,
        Oracle(scenes, changed_labels, changed_margins),
        pretrained=base,
        history=history,
        query_order=order[:1],
        budget=1,
        snapshot_counts=(0, 1),
        inference_backend=inference_backend,
    )
    assert oracle.queried == order
    assert full["prediction_snapshots"][:2] == short["prediction_snapshots"]
    assert [s["observed_indices"] for s in full["prediction_snapshots"]] == [[], [3], order]
    assert [s["training_count"] for s in full["prediction_snapshots"]] == [2, 3, 5]
    assert (tmp_path / "partial.json").exists()

    checkpoint = select(
        scenes,
        Oracle(scenes, labels, margins),
        pretrained=base,
        history=history,
        query_order=order,
        budget=3,
        snapshot_counts=(3,),
        profile_path=tmp_path / "stages.json",
        inference_backend=inference_backend,
    )
    assert checkpoint["prediction_snapshots"] == full["prediction_snapshots"][-1:]
    prior_oracle = Oracle(scenes, changed_labels, changed_margins)
    prior = select(
        scenes,
        prior_oracle,
        pretrained=base,
        history=history,
        query_order=[],
        budget=0,
        snapshot_counts=(0,),
        inference_backend=inference_backend,
    )
    assert prior_oracle.queried == []
    assert prior["prediction_snapshots"] == full["prediction_snapshots"][:1]
    assert (tmp_path / "stages.json").exists()


@pytest.mark.parametrize("inference_backend", ["dense", "grouped"])
def test_concurrent_checkpoint_children_keep_disclosed_feedback_separate(inference_backend):
    from experiments.query import PaidExecution, QueryLedger, run_isolated

    scenes = sample("car_following", 8, "new", replicate=90)
    x = features(scenes)
    base = SharedPrivateKernel(3, deep=False).double()
    history = {"x": x[:2], "collision": [False, True], "rss_margin": [-0.4, 0.2]}
    order = [3, 1, 6]

    def run(group):
        def execute(scene):
            return PaidExecution(
                QueryFeedback(
                    scene.scenario_id,
                    bool(group),
                    SafetyMeasurements(1, 1, rss_margin=0.2 if group else -0.4),
                    0,
                    measurement_contract(),
                ),
                [],
            )

        ledger = QueryLedger(scenes, 3, execute, allow_registration=False)
        return run_isolated(
            select,
            {
                "candidates": scenes,
                "pretrained": base,
                "history": history,
                "query_order": order,
                "budget": 3,
                "snapshot_counts": (0, 3),
                "threads": 4 if group else 1,
                "inference_backend": inference_backend,
            },
            ledger,
            wall_limit_s=60,
            memory_limit_bytes=2 * 1024**3,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run, (0, 1)))
    for group, result in enumerate((first, second)):
        assert result["torch_threads"] == (4 if group else 1)
        assert result["selected_indices"] == order
        assert all(q["feedback"]["collision"] == bool(group) for q in result["queries"])
    assert first["prediction_snapshots"][0] == second["prediction_snapshots"][0]
    assert (
        first["prediction_snapshots"][1]["probabilities"]
        != second["prediction_snapshots"][1]["probabilities"]
    )


def test_four_threads_preserve_cold_fit_probabilities_and_scalar_parameters():
    scenes = sample("car_following", 300, "new", replicate=90)
    x = features(scenes)
    base = SharedPrivateKernel(3, deep=False).double()
    history = {
        "x": x[:256],
        "collision": np.arange(256) % 3 == 0,
        "rss_margin": np.linspace(-0.4, 0.3, 256),
    }
    labels, margins = np.arange(300) % 2 == 0, np.linspace(-0.5, 0.4, 300)
    order = [280, 289, 295]
    results = []
    try:
        for threads in (1, 4):
            result = select(
                scenes,
                Oracle(scenes, labels, margins),
                pretrained=base,
                history=history,
                query_order=order,
                budget=3,
                snapshot_counts=(0, 3),
                threads=threads,
            )
            assert result["torch_threads"] == threads
            results.append(result)
        for first, second in zip(
            results[0]["prediction_snapshots"], results[1]["prediction_snapshots"]
        ):
            np.testing.assert_allclose(
                first["probabilities"], second["probabilities"], rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                list(first["target_parameters"].values()),
                list(second["target_parameters"].values()),
                rtol=1e-10,
                atol=1e-10,
            )
    finally:
        torch.set_num_threads(1)
