import json
import pickle

import numpy as np
import pytest
import torch

from experiments.domain import Domain
from experiments.failure_metrics import parameter_distances
from experiments.rss import measurement_contract
from experiments.scenarios import DOMAINS, features, sample
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from testing_methods.history_transfer.selection import select


@pytest.mark.parametrize("kind", DOMAINS)
def test_explicit_domain_matches_retained_features_and_gower_distance(kind):
    existing = DOMAINS[kind]
    domain = Domain(existing.bounds, existing.quantization, existing.categories)
    scenes = sample(kind, 7, "history", replicate=22)
    np.testing.assert_array_equal(domain.features(scenes), features(scenes))
    np.testing.assert_array_equal(
        domain.parameter_distances(scenes[:3], scenes[3:]),
        parameter_distances(scenes[:3], scenes[3:]),
    )
    assert domain.dimension == existing.dimension


def braking_domain():
    return Domain(
        {"gap_m": (12, 40), "deceleration_mps2": (2, 6)},
        {"gap_m": 1, "deceleration_mps2": 0.5},
        {},
        {"ego_speed_mps": 50 / 3.6, "closing_speed_mps": 0, "event_time_s": 1, "brake_ramp_s": 1},
    )


def test_fixed_parameters_do_not_add_features_or_distance_dimensions():
    domain = braking_domain()
    low = Scenario("low", "front_brake", {"gap_m": 12, "deceleration_mps2": 2})
    high = Scenario("high", "front_brake", {"gap_m": 40, "deceleration_mps2": 6})
    assert domain.dimension == 2
    np.testing.assert_array_equal(domain.features([low, high]), [[0, 0], [1, 1]])
    np.testing.assert_array_equal(domain.parameter_distances([low], [high]), [[1]])
    parameters, categories = domain.execution_values(low)
    assert parameters == {**domain.fixed_parameters, **low.parameters} and categories == {}
    with pytest.raises(ValueError, match="exactly the free"):
        domain.features([Scenario("redundant", "front_brake", parameters)])


def test_domain_serialization_binds_feature_order_and_is_spawn_serializable():
    domain = braking_domain()
    restored = Domain.from_dict(json.loads(json.dumps(domain.to_dict(), sort_keys=True)))
    assert restored == domain and pickle.loads(pickle.dumps(domain)) == domain
    reversed_domain = Domain(
        dict(reversed(list(domain.bounds.items()))),
        domain.quantization,
        {},
        domain.fixed_parameters,
    )
    assert json.dumps(reversed_domain.to_dict(), sort_keys=True) != json.dumps(
        domain.to_dict(), sort_keys=True
    )
    broken = domain.to_dict()
    broken["bounds"].append(broken["bounds"][0])
    with pytest.raises(ValueError, match="unique"):
        Domain.from_dict(broken)


def test_category_distance_counts_the_attribute_once_instead_of_its_one_hot_width():
    domain = Domain(
        {"gap": (0, 10)},
        {"gap": 1},
        {"weather": ("dry", "wet", "snow")},
        fixed_categories={"side": "left"},
    )
    a = Scenario("a", "example", {"gap": 0}, {"weather": "dry"})
    b = Scenario("b", "example", {"gap": 0}, {"weather": "snow"})
    assert domain.dimension == 4
    np.testing.assert_array_equal(domain.parameter_distances([a], [b]), [[0.5]])
    assert domain.execution_values(a)[1] == {"side": "left", "weather": "dry"}


@pytest.mark.parametrize(
    "change",
    [
        {"quantization": {}},
        {"bounds": {"gap": (1, 1)}},
        {"quantization": {"gap": -1}},
        {"fixed_parameters": {"gap": 5}},
        {"categories": {"side": ("left", "left")}},
        {"categories": {"side": ("left", "right")}, "fixed_categories": {"side": "left"}},
    ],
)
def test_invalid_parameter_definitions_are_rejected(change):
    with pytest.raises(ValueError):
        Domain(
            **{"bounds": {"gap": (0, 10)}, "quantization": {"gap": 1}, "categories": {}, **change}
        )


def test_existing_raw_joint_selector_accepts_a_two_dimensional_feature_definition():
    torch.set_num_threads(1)
    domain = braking_domain()
    scenes = [
        Scenario(str(i), "front_brake", {"gap_m": 12 + 7 * i, "deceleration_mps2": 2 + i})
        for i in range(5)
    ]

    class Oracle:
        def query(self, index, *, purpose):
            return QueryFeedback(
                scenes[index].scenario_id,
                index < 2,
                SafetyMeasurements(5, 1, rss_margin=-0.7 if index < 2 else 0.4),
                0,
                measurement_contract(),
            )

    result = select(scenes, Oracle(), feature_fn=domain.features, initial_indices=[0], budget=3)
    assert len(set(result["selected_indices"])) == result["actual_queries"] == 3
    assert result["representation"] == "normalized_parameters"
