import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.history_transfer.inference import mixture_weights
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.reliability import local_weights
from testing_methods.history_transfer.selection import select


def test_local_weights_recover_global_with_no_evidence_uniform_evidence_or_no_neighbours():
    scores = torch.tensor([0.2, -0.6], dtype=torch.float64)
    locations = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    expected = mixture_weights(scores)[:, None].expand(-1, 2)
    actual = local_weights(scores, locations, locations[:0], [], 0.3)
    torch.testing.assert_close(actual, expected)
    likelihood = [scores, scores]
    actual = local_weights(scores, locations, locations, likelihood, 0.3)
    torch.testing.assert_close(actual, expected)
    actual = local_weights(scores, locations + 1000, locations, [scores, -scores], 0.01)
    torch.testing.assert_close(actual, expected)


def test_local_weights_reverse_source_preference_and_stream_only_cross_distances(monkeypatch):
    scores = torch.zeros(2, dtype=torch.float64)
    locations = torch.linspace(0, 1, 21, dtype=torch.float64)[:, None]
    paid = torch.tensor([[0.0]] * 20 + [[1.0]] * 20, dtype=torch.float64)
    likelihood = [torch.tensor(v, dtype=torch.float64) for v in [[-0.1, -2.0]] * 20]
    likelihood += [torch.tensor(v, dtype=torch.float64) for v in [[-2.0, -0.1]] * 20]
    expected = local_weights(scores, locations, paid, likelihood, 0.1)
    assert expected[0, 0] > expected[1, 0]
    assert expected[1, -1] > expected[0, -1]
    torch.testing.assert_close(expected.sum(0), torch.ones(21, dtype=torch.float64))
    assert (expected >= 0).all()
    assert expected[-1].tolist() == [0.1] * 21
    cdist = torch.cdist

    def observed(left, right):
        assert len(left) <= 3
        assert len(right) == len(paid)
        return cdist(left, right)

    monkeypatch.setattr(torch, "cdist", observed)
    actual = local_weights(scores, locations, paid, likelihood, 0.1, batch_size=3)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_local_selector_uses_paid_feedback_only_and_preserves_batch_selection_sequence():
    torch.set_num_threads(1)
    torch.manual_seed(1301)
    scenes = sample("car_following", 13, "history")
    model = JointResponseKernel(features(scenes).shape[1]).double()
    sources = {
        "names": ["a", "b"],
        "weights": np.array([0.5, 0.5]),
        "means": np.random.default_rng(1).normal(size=(2, len(scenes), 2)),
        "blocks": np.tile(np.array([[0.4, -0.1], [-0.1, 0.2]]), (2, len(scenes), 1, 1)),
    }

    class Oracle:
        def __init__(self, changed=None):
            self.changed = changed

        def query(self, index, **kwargs):
            hit = index % 3 == 0 if self.changed != index else index % 3 != 0
            return QueryFeedback(
                scenario_id=scenes[index].scenario_id,
                collision=hit,
                safety=SafetyMeasurements(5, 1, rss_margin=-0.4 if hit else 0.3),
                execution_s=0,
                rss_contract=measurement_contract(),
            )

    arguments = dict(
        candidates=scenes,
        pretrained=model,
        sources=sources,
        initial_indices=[0, 1],
        budget=6,
        local_reliability=True,
        snapshot_counts=(2, 5, 6),
    )
    a = select(oracle=Oracle(), batch_size=3, **arguments)
    b = select(oracle=Oracle(), batch_size=8, **arguments)
    assert a["selected_indices"] == b["selected_indices"]
    for sa, sb in zip(a["prediction_snapshots"], b["prediction_snapshots"]):
        np.testing.assert_allclose(sa["probabilities"], sb["probabilities"], atol=1e-10)
    future = a["selected_indices"][4]
    changed = select(oracle=Oracle(changed=future), batch_size=3, **arguments)
    assert changed["selected_indices"][:5] == a["selected_indices"][:5]
    for left, right in zip(a["queries"][:5], changed["queries"][:5]):
        assert left["prequery_probability"] == pytest.approx(right["prequery_probability"])
        assert left["prequery_component_weights"] == pytest.approx(
            right["prequery_component_weights"]
        )
