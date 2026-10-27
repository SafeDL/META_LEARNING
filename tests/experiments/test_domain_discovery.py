from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.common.gp_classifier import ClassificationKernel
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.domain_build import historical_posterior
from testing_methods.history_transfer.domain_discovery import aggregate_history, select
from testing_methods.history_transfer.joint_response import JointResponseKernel


def sources(size):
    return {
        "names": ["a", "b"],
        "weights": np.array([0.25, 0.75]),
        "means": np.stack((np.linspace(-1, 1, size), np.linspace(0.5, -0.5, size))),
        "covariances": np.stack((np.eye(size), 2 * np.eye(size))),
    }


def test_teacher_preserves_mixture_mean_and_total_covariance():
    raw = sources(3)
    teacher = aggregate_history(raw)
    mean = raw["weights"] @ raw["means"]
    second = sum(
        w * (c + np.outer(m, m))
        for w, c, m in zip(raw["weights"], raw["covariances"], raw["means"])
    )
    assert teacher["means"][0] == pytest.approx(mean)
    assert teacher["covariances"][0] == pytest.approx(second - np.outer(mean, mean))
    assert np.linalg.eigvalsh(teacher["covariances"][0]).min() > 0


@pytest.mark.parametrize("mode", ["H0F0", "H1F0"])
def test_f0_never_conditions_on_initial_or_later_paid_labels(mode):
    scenes = sample("cut_in", 12, "development")
    prior = sources(len(scenes))
    prior = aggregate_history(prior) if mode.startswith("H0") else prior
    arguments = {
        "candidates": scenes,
        "pretrained": ClassificationKernel(features(scenes).shape[1]),
        "sources": prior,
        "mode": mode,
        "initial_indices": [4, 2],
        "budget": 7,
        "device": "cpu",
    }
    outputs = []
    for collision in (False, True):

        def execute(scene):
            return PaidExecution(
                QueryFeedback(scene.scenario_id, collision, SafetyMeasurements(0, 0), 1), []
            )

        outputs.append(run_isolated(select, arguments, QueryLedger(scenes, 7, execute)))
    assert outputs[0]["selected_indices"] == outputs[1]["selected_indices"]
    assert [r["prequery_probability"] for r in outputs[0]["queries"]] == pytest.approx(
        [r["prequery_probability"] for r in outputs[1]["queries"]]
    )
    for output in outputs:
        for row in output["queries"]:
            assert row["conditioning_queries"] == 0
            assert row["prequery_log_likelihood"] is None
            assert row["component_weights_after_query"] == pytest.approx(
                row["prequery_component_weights"]
            )


def test_f1_source_permutation_preserves_queries_and_updates_only_prequery_likelihood():
    scenes = sample("cut_out", 12, "development")
    base = ClassificationKernel(features(scenes).shape[1])
    raw = sources(len(scenes))
    outputs = []
    for order in ([0, 1], [1, 0]):
        prior = {
            key: np.asarray(value)[order].tolist() if key == "names" else np.asarray(value)[order]
            for key, value in raw.items()
        }

        def execute(scene):
            return PaidExecution(
                QueryFeedback(
                    scene.scenario_id, scenes.index(scene) % 2 == 0, SafetyMeasurements(999, 999), 1
                ),
                [],
            )

        outputs.append(
            run_isolated(
                select,
                {
                    "candidates": scenes,
                    "pretrained": base,
                    "sources": prior,
                    "mode": "H1F1",
                    "initial_indices": [4, 2],
                    "budget": 7,
                    "device": "cpu",
                },
                QueryLedger(scenes, 7, execute),
            )
        )
    assert outputs[0]["selected_indices"] == outputs[1]["selected_indices"]
    for step, row in enumerate(outputs[0]["queries"]):
        assert row["conditioning_queries"] == step
        probability = np.asarray(row["prequery_component_probabilities"][:-1])
        likelihood = probability if row["feedback"]["collision"] else 1 - probability
        assert row["prequery_log_likelihood"] == pytest.approx(np.log(likelihood))
        assert row["component_weights_after_query"][-1] == pytest.approx(0.1)
        assert max(d["gradient_max"] for d in row["conditional_modes"]) <= 1e-10
    for a, b in zip(outputs[0]["queries"], outputs[1]["queries"]):
        assert a["prequery_probability"] == pytest.approx(b["prequery_probability"], abs=1e-12)


def test_historical_posterior_converges_and_preserves_positive_covariance():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    scenes = sample("cut_in", 8, "history")
    archive = ResponseArchive(
        "a",
        "rule",
        "a",
        "history",
        scenes,
        np.arange(8) % 3 == 0,
        np.column_stack((np.linspace(0, 20, 8), np.ones(8))),
    )
    model = JointResponseKernel(features(scenes).shape[1])
    mean, covariance, diagnostics = historical_posterior(model, archive, archive.x[:3])
    assert mean.shape == (3,) and covariance.shape == (3, 3)
    assert np.isfinite(mean).all() and np.linalg.eigvalsh(covariance).min() > -1e-8
    assert diagnostics["gradient_max"] <= 1e-10
    altered = replace(archive, collision=~archive.collision)
    changed, _, _ = historical_posterior(model, altered, archive.x[:3])
    assert not np.allclose(mean, changed)


def test_eight_source_diagonal_interaction_inference_is_invariant_to_source_order():
    torch.set_num_threads(1)
    torch.manual_seed(1301)
    scenes = sample("cut_in_interaction", 24, "development")
    base = ClassificationKernel(features(scenes).shape[1], deep=True).double()
    rng = np.random.default_rng(1319)
    raw = {
        "names": [f"historical_fixture_{i}" for i in range(8)],
        "weights": np.arange(1, 9, dtype=float) / 36,
        "means": rng.normal(0, 2, size=(8, len(scenes))),
        "covariances": np.stack([np.diag(rng.uniform(0.1, 2, size=len(scenes))) for _ in range(8)]),
    }

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id, index % 3 == 0, SafetyMeasurements(1, 10), 0
            )

    outputs = []
    orders = (np.arange(8), np.array([3, 5, 0, 7, 1, 6, 2, 4]))
    for order in orders:
        prior = {
            key: np.asarray(value)[order].tolist() if key == "names" else np.asarray(value)[order]
            for key, value in raw.items()
        }
        outputs.append(
            select(
                scenes,
                Oracle(),
                pretrained=base,
                sources=prior,
                mode="H1F1",
                initial_indices=list(range(8)),
                budget=12,
                device="cpu",
            )
        )
    assert outputs[0]["selected_indices"] == outputs[1]["selected_indices"]
    inverse = np.r_[np.argsort(orders[1]), 8, 9]
    for original, permuted in zip(outputs[0]["queries"], outputs[1]["queries"]):
        assert original["prequery_probability"] == pytest.approx(
            permuted["prequery_probability"], abs=1e-12
        )
        for key in (
            "prequery_component_probabilities",
            "prequery_component_weights",
            "component_weights_after_query",
        ):
            assert original[key] == pytest.approx(np.asarray(permuted[key])[inverse], abs=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA precision comparison")
def test_double_feature_adaptation_matches_cpu_and_cuda():
    from testing_methods.common.gp_classifier import fit_target

    torch.set_num_threads(1)
    torch.manual_seed(11)
    x = features(sample("cut_in", 32, "development"))
    base = ClassificationKernel(x.shape[1], deep=True).double()
    gpu = ClassificationKernel(x.shape[1], deep=True).double().cuda()
    gpu.load_state_dict(base.state_dict())
    labels = np.arange(len(x)) % 4 == 0
    for model in (base, gpu):
        fit_target(model, x, labels, steps=10, learning_rate=0.003)
    with torch.no_grad():
        assert base.covariance(x).numpy() == pytest.approx(
            gpu.covariance(x).cpu().numpy(), abs=1e-7
        )
