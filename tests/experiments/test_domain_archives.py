from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.domain import Domain
from experiments.failure_metrics import summarize
from experiments.history import build_source, update_shared_model
from experiments.rss import measurement_contract
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from experiments.storage import load_archive, save_archive
from testing_methods.baselines.paired_system_active_evaluation.method import build_proxy
from testing_methods.common.response import ResponseArchive, archive_fingerprints, shared_domain
from testing_methods.history_transfer.inference import adapted_model, predict_components
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import Config, load_model, specification, train


def domain():
    return Domain(
        {"gap_m": (12, 40), "deceleration_mps2": (2, 6)},
        {"gap_m": 1, "deceleration_mps2": 0.1},
        {},
        {"ego_speed_mps": 50 / 3.6},
    )


def archive(name="a", definition=None):
    scenes = [
        Scenario(str(i), "front_brake", {"gap_m": 12 + 3 * i, "deceleration_mps2": 2 + 0.4 * i})
        for i in range(8)
    ]
    y = np.arange(8) % 3 == 0
    return ResponseArchive(
        name,
        "rule",
        name,
        "history",
        scenes,
        y,
        np.column_stack((np.where(y, 0, 5), np.ones(8))),
        rss_margin=np.where(y, -0.6, 0.3),
        rss_contract=measurement_contract(),
        domain=definition or domain(),
    )


class Oracle:
    def __init__(self, scenes):
        self.scenes, self.calls = scenes, []

    def query(self, index, *, purpose):
        self.calls.append(index)
        collision = index % 3 == 0
        return QueryFeedback(
            self.scenes[index].scenario_id,
            collision,
            SafetyMeasurements(0 if collision else 5, 1, rss_margin=-0.6 if collision else 0.3),
            0,
            measurement_contract(),
        )


def test_archive_roundtrip_preserves_definition_and_fingerprints_bind_fixed_values_and_order(
    tmp_path,
):
    a = archive()
    path = tmp_path / "archive.npz"
    save_archive(path, a, provenance=a.provenance)
    restored = load_archive(path)
    assert restored.domain.to_dict() == a.domain.to_dict()
    np.testing.assert_array_equal(restored.x, a.x)
    assert archive_fingerprints([restored]) == archive_fingerprints([a])
    fixed = replace(a.domain, fixed_parameters={"ego_speed_mps": 12})
    ordered = replace(a.domain, bounds=dict(reversed(list(a.domain.bounds.items()))))
    for changed in [fixed, ordered]:
        other = replace(a, domain=changed)
        assert archive_fingerprints([other]) != archive_fingerprints([a])
        with pytest.raises(ValueError, match="different parameter definitions"):
            shared_domain([a, other])


def test_source_cache_rejects_equal_dimension_but_different_parameter_range(tmp_path):
    torch.set_num_threads(1)
    archives = [archive("a"), archive("b")]
    model = SharedPrivateKernel(2).double()
    path = tmp_path / "source.npz"
    original = build_source(model, archives, archives[0].scenarios, cache_path=path)
    restored = build_source(model, archives, archives[0].scenarios, cache_path=path)
    np.testing.assert_array_equal(restored["means"], original["means"])
    changed = replace(domain(), bounds={"gap_m": (10, 45), "deceleration_mps2": (2, 6)})
    changed_archives = [replace(a, domain=changed) for a in archives]
    with pytest.raises(ValueError, match="Source cache differs"):
        build_source(model, changed_archives, archives[0].scenarios, cache_path=path)
    with pytest.raises(ValueError, match="different parameter definitions"):
        build_source(model, archives, archives[0].scenarios, domain=changed)


def test_history_proxy_uses_its_archive_definition_and_rejects_another_normalizer():
    archives = [archive("a"), archive("b")]
    scenes = archives[0].scenarios
    implicit, record = build_proxy(scenes, archives)
    explicit, _ = build_proxy(scenes, archives, feature_fn=domain().features)
    np.testing.assert_array_equal(implicit, explicit)
    assert record["parameter_definition"] == domain().to_dict()
    changed = replace(domain(), bounds={"gap_m": (10, 45), "deceleration_mps2": (2, 6)})
    with pytest.raises(ValueError, match="Proxy features differ"):
        build_proxy(scenes, archives, feature_fn=changed.features)


def test_two_stage_model_source_and_selector_share_registered_two_dimensional_definition(tmp_path):
    torch.set_num_threads(1)
    archives = [archive("a"), archive("b")]
    config = Config(steps=1, initial_count=2, budget=5, historical_context=4)
    train(archives, tmp_path, config=config, kernel_type=SharedPrivateKernel)
    model, _ = load_model(tmp_path / "target_kernel.pt", kernel_type=SharedPrivateKernel)
    assert model.construction_specification["parameter_definition"] == domain().to_dict()
    sources = build_source(model, archives, archives[0].scenarios)
    oracle = Oracle(archives[0].scenarios)
    result = select(
        archives[0].scenarios,
        oracle,
        pretrained=model,
        sources=sources,
        initial_indices=[0, 1],
        budget=3,
    )
    assert result["response_kernel"] == "SharedPrivateKernel"
    assert len(set(oracle.calls)) == 3
    changed = replace(domain(), fixed_parameters={"ego_speed_mps": 12})
    with pytest.raises(ValueError, match="Target features differ"):
        select(
            archives[0].scenarios,
            Oracle(archives[0].scenarios),
            pretrained=model,
            sources=sources,
            feature_fn=changed.features,
            initial_indices=[0, 1],
            budget=3,
        )
    with pytest.raises(ValueError, match="another parameter definition"):
        build_source(model, [replace(a, domain=changed) for a in archives], archives[0].scenarios)
    with pytest.raises(ValueError, match="different parameter definitions"):
        specification([archives[0], replace(archives[1], domain=changed)], None, config)


def test_between_task_publication_preserves_shared_private_type_and_parameter_definition(tmp_path):
    torch.set_num_threads(1)
    archives = [archive("a"), archive("b")]
    config = Config(steps=1, initial_count=2, budget=5, historical_context=4)
    initial_path = tmp_path / "initial"
    train(archives, initial_path, config=config, kernel_type=SharedPrivateKernel)
    current = tuple(
        load_model(initial_path / f"{part}.pt", kernel_type=SharedPrivateKernel)[0]
        for part in ("target_kernel", "historical_encoder")
    )
    published, record = update_shared_model(
        tmp_path / "published",
        [*archives, archive("c")],
        current,
        new_source_ids=["c"],
        config=config,
    )
    assert record["accepted"]
    assert all(type(m) is SharedPrivateKernel for m in published)
    assert all(
        m.construction_specification["parameter_definition"] == domain().to_dict()
        for m in published
    )


def test_no_history_shared_private_control_is_one_raw_gp_with_default_structure_parameters():
    torch.set_num_threads(1)
    scenes = archive().scenarios
    result = select(
        scenes,
        Oracle(scenes),
        initial_indices=[0, 1],
        budget=3,
        feature_fn=domain().features,
        raw_kernel_type=SharedPrivateKernel,
        snapshot_counts=(0, 2, 3),
    )
    assert result["response_kernel"] == "SharedPrivateKernel"
    assert result["component_names"] == ["raw_target"]
    assert result["physical_mass"] == 1 and not result["global_reliability"]
    assert result["representation"] == "normalized_parameters"
    x = domain().features(scenes)
    base = SharedPrivateKernel(2, deep=False).double()
    fitted = adapted_model(base, x[:2], [True, False], [-0.6, 0.3], learning_rate=0.01)
    with torch.no_grad():
        reference, _ = predict_components(
            fitted, None, x, [], base.point_mean(1)[0], [0, 1], [True, False], [-0.6, 0.3]
        )
    snapshot = next(s for s in result["prediction_snapshots"] if s["paid_queries"] == 2)
    np.testing.assert_allclose(snapshot["probabilities"], reference[0], atol=1e-12, rtol=0)
    for name in ("raw_shared_fraction", "log_collision_lengthscale", "log_rss_lengthscale"):
        assert torch.equal(getattr(base, name), getattr(fitted, name))


def test_metrics_use_registered_free_parameters_and_keep_fixed_parameters_out():
    definition = domain()
    scenes = [
        Scenario(
            str(i),
            "front_brake",
            {name: lo + f * (hi - lo) for name, (lo, hi) in definition.bounds.items()},
        )
        for i, f in enumerate([0, 0.5, 1, 0.25])
    ]
    labels, queries, traces, ledger = [True, True, True, False], [], [], []
    for index in [0, 3, 2, 1]:
        f = QueryFeedback(scenes[index].scenario_id, labels[index], SafetyMeasurements(0, 0), 0)
        queries.append({"index": index, "feedback": f.to_dict()})
        traces.append([{"collision": labels[index]}])
        ledger.append({"index": index, "feedback": f.to_dict(), "status": "complete", "wall_s": 0})
    result = summarize(
        scenes, labels, queries, traces, initial_count=1, query_ledger=ledger, domain=definition
    )
    assert result["unique_collisions"] == 3 and result["collision_recall"] == 1
    assert result["discovery_auc"] == 1.75
    assert result["pair_distance"] == pytest.approx(2 / 3)
    assert result["nearest_neighbor_distance"] == 0.5
