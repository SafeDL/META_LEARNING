import numpy as np
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.history_transfer.inference import adapted_model, predict_components
from testing_methods.history_transfer.model import JointResponseKernel
from testing_methods.history_transfer.selection import select


def test_encoded_kernel_preserves_covariance_and_scalar_gradients():
    torch.manual_seed(1301)
    x = features(sample("car_following", 13, "history"))
    model = JointResponseKernel(3).double()
    z = model.features(x).detach()
    actual = model.kernel_from_features(z[:7], z[7:])
    scalar = model.covariance(x[:7], x[7:])
    expected = torch.einsum("ij,ab->iajb", scalar, model.output_covariance()).reshape(14, 12)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    parameters = [
        model.log_lengthscale,
        model.log_scale,
        model.log_margin_scale,
        model.raw_correlation,
    ]
    gradients = torch.autograd.grad(actual.square().sum(), parameters)
    reference = torch.autograd.grad(expected.square().sum(), parameters)
    for cached, uncached in zip(gradients, reference):
        torch.testing.assert_close(cached, uncached, rtol=1e-12, atol=1e-12)


def test_cached_features_preserve_adaptation_and_predictions():
    torch.set_num_threads(1)
    torch.manual_seed(1301)
    x = features(sample("front_brake", 13, "history"))
    model = JointResponseKernel(6).double()
    raw = JointResponseKernel(6, deep=False).double()
    z, raw_z = model.features(x).detach(), raw.features(x)
    paid, labels, margins = [0, 4, 6], [False, True, False], [0.3, -0.9, 0.2]
    cached = adapted_model(model, x[paid], labels, margins, encoded=z[paid])
    reference = adapted_model(model, x[paid], labels, margins)
    for key, value in cached.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=1e-12, atol=1e-12)
    common = (cached, raw, x, [], model.point_mean(1)[0], paid, labels, margins)
    actual, _ = predict_components(*common, encoded_target=z, encoded_physical=raw_z, batch_size=4)
    expected, _ = predict_components(*common, batch_size=9)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_online_shared_features_are_computed_once_for_all_queries(monkeypatch):
    scenes = sample("car_following", 12, "history")
    calls = []
    original = JointResponseKernel.features

    def counted(model, x):
        if model.deep:
            calls.append(len(x))
        return original(model, x)

    class Oracle:
        def query(self, index, **kwargs):
            collision = index % 3 == 0
            return QueryFeedback(
                scenes[index].scenario_id,
                collision,
                SafetyMeasurements(0 if collision else 8, 1, rss_margin=-0.8 if collision else 0.3),
                0.1,
                rss_contract=measurement_contract(),
            )

    torch.manual_seed(1301)
    model = JointResponseKernel(3).double()
    monkeypatch.setattr(JointResponseKernel, "features", counted)
    result = select(
        scenes, Oracle(), pretrained=model, initial_indices=[0, 2], budget=5, batch_size=3
    )
    assert calls == [len(scenes)]
    assert len(set(result["selected_indices"])) == 5
    assert np.isfinite([r["prequery_probability"] for r in result["queries"]]).all()
