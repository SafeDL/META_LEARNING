"""Configured fitting must reproduce development fits and preserve information boundaries."""

import copy

import numpy as np
import pytest
import torch

from experiments.rss import measurement_contract
from experiments.scenarios import features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.transfer_settings import native_settings
from testing_methods.baselines.response_baselines.pooled_gp import predict_pooled
from testing_methods.history_transfer.adaptation import FitSettings, adapted_online
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


class Oracle:
    def __init__(self, scenes, changed=False):
        self.scenes, self.changed = scenes, changed

    def query(self, index, *, purpose):
        return QueryFeedback(
            self.scenes[index].scenario_id,
            bool(index == 1 and not self.changed),
            SafetyMeasurements(1, 1, rss_margin=(-0.4 if index == 1 else 0.5)),
            0,
            measurement_contract(),
        )


@pytest.mark.parametrize("with_history_model", [False, True])
def test_formal_native_settings_reproduce_the_original_default_path(with_history_model):
    scenes = sample("car_following", 6, "new", replicate=512)
    options = dict(initial_indices=[0], budget=2, query_order=[0, 1], snapshot_counts=(0, 1, 2))
    if with_history_model:
        options["pretrained"] = SharedPrivateKernel(features(scenes).shape[1]).double()
    else:
        options["raw_kernel_type"] = SharedPrivateKernel
    original = select(scenes, Oracle(scenes), **options)
    explicit = select(scenes, Oracle(scenes), online_settings=native_settings().online, **options)
    for expected, actual in zip(original["prediction_snapshots"], explicit["prediction_snapshots"]):
        assert expected["target_parameters"] == actual["target_parameters"]
        assert expected["physical_parameters"] == actual["physical_parameters"]
        np.testing.assert_allclose(
            expected["probabilities"], actual["probabilities"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            expected["component_weights"], actual["component_weights"], atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_formal_pooled_settings_reproduce_the_original_default_path(backend):
    torch.set_num_threads(1)
    base = SharedPrivateKernel(1).double()
    x = np.array([[0.1], [0.3], [0.7], [0.9]])
    labels, margins = np.array([False, False, True, True]), np.array([0.4, 0.2, -0.2, -0.5])
    default = predict_pooled(base, x, labels, margins, x, inference_backend=backend)
    explicit = predict_pooled(
        base,
        x,
        labels,
        margins,
        x,
        inference_backend=backend,
        fit_settings=native_settings().pooled,
    )
    np.testing.assert_allclose(default[0], explicit[0], atol=1e-12, rtol=0)
    assert default[1] == explicit[1]


def test_raw_branch_is_identical_in_target_only_and_ours_with_native_settings():
    scenes = sample("car_following", 6, "new", replicate=512)
    settings = native_settings().online
    options = dict(
        initial_indices=[0],
        budget=2,
        query_order=[0, 1],
        snapshot_counts=(0, 1, 2),
        online_settings=settings,
    )
    raw = select(scenes, Oracle(scenes), raw_kernel_type=SharedPrivateKernel, **options)
    base = SharedPrivateKernel(features(scenes).shape[1]).double()
    ours = select(scenes, Oracle(scenes), pretrained=base, **options)
    for target, full in zip(raw["prediction_snapshots"], ours["prediction_snapshots"]):
        assert target["target_parameters"] == full["physical_parameters"]
        assert target["target_fit"] == full["physical_fit"]
    assert raw["online_settings"] == settings.to_dict()
    changed = select(
        scenes, Oracle(scenes, changed=True), raw_kernel_type=SharedPrivateKernel, **options
    )
    assert raw["prediction_snapshots"][:2] == changed["prediction_snapshots"][:2]
    assert (
        raw["prediction_snapshots"][2]["probabilities"]
        != changed["prediction_snapshots"][2]["probabilities"]
    )


def test_pooled_settings_reach_both_backends_without_changing_the_frozen_base():
    torch.set_num_threads(1)
    x = np.array([[0.1], [0.3], [0.7], [0.9]])
    labels, margins = np.array([False, False, True, True]), np.array([0.4, 0.2, -0.2, -0.5])
    base = SharedPrivateKernel(1).double()
    before = copy.deepcopy(base.state_dict())
    for p in base.parameters():
        p.requires_grad_(False)
    settings = native_settings().pooled
    dense, parameters, _ = predict_pooled(base, x, labels, margins, x, fit_settings=settings)
    grouped, other, _ = predict_pooled(
        base, x, labels, margins, x, inference_backend="grouped", fit_settings=settings
    )
    np.testing.assert_allclose(dense, grouped, atol=1e-10, rtol=0)
    assert set(parameters) == set(other) == set(settings.parameters)
    for name in parameters:
        assert parameters[name] == pytest.approx(other[name], abs=1e-10)
    assert all(torch.equal(value, before[name]) for name, value in base.state_dict().items())


def test_configured_fitting_rejects_unknown_parameters_and_keeps_empty_prior():
    base = SharedPrivateKernel(1, deep=False).double()
    settings = FitSettings(10, 0.01)
    empty = adapted_online(base, np.empty((0, 1)), [], [], settings)
    assert all(
        torch.equal(value, base.state_dict()[name]) for name, value in empty.state_dict().items()
    )
    with pytest.raises(ValueError, match="Unknown online GP parameter"):
        adapted_online(base, np.empty((0, 1)), [], [], FitSettings(10, 0.01, ("unknown",)))
