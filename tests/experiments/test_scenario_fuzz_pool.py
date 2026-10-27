import numpy as np
import pytest
import torch

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import DOMAINS, sample, validate
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.scenario_fuzz.pool import Config, mutate, run


class ConstantRisk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, data):
        return self.logit.expand(len(data.weather_attr))


def test_neighbor_mutates_all_attributes_within_five_public_precision_steps():
    generator = np.random.default_rng(11)
    for kind in ("cut_in", "cut_out"):
        scene = sample(kind, 1, "development")[0]
        changed = set()
        for _ in range(100):
            proposal = mutate(scene, generator, local=True)
            validate(proposal)
            for name, precision in DOMAINS[kind].quantization.items():
                difference = (proposal.parameters[name] - scene.parameters[name]) / precision
                assert abs(difference) <= 5 + 1e-9
                assert difference == pytest.approx(round(difference))
                if difference != 0:
                    changed.add(name)
        assert changed == set(DOMAINS[kind].bounds)


def test_fixed_model_filter_completion_and_native_score_guide_paid_successor():
    scenes = sample("cut_in", 24, "development")
    model = ConstantRisk()

    def execute(scene):
        return PaidExecution(
            QueryFeedback(
                scene.scenario_id,
                False,
                SafetyMeasurements(10, 10, driving_quality_score=-float(scenes.index(scene))),
                1,
            ),
            [],
        )

    result = run_isolated(
        run,
        {
            "candidates": scenes,
            "classifier": model,
            "initial_indices": [4, 2, 8],
            "budget": 16,
            "config": Config(mutants=12),
        },
        QueryLedger(scenes, 16, execute),
    )
    assert len(set(result["selected_indices"])) == result["actual_queries"] == 16
    assert result["selected_indices"][:3] == [4, 2, 8]
    assert result["threshold_fill_queries"] == 13
    assert result["online_classifier_updates"] == 0
    assert model.logit.item() == -1.0
    first = [r for r in result["queries"] if r["campaign"] == 1 and r["cycle"] == 1]
    second = [r for r in result["queries"] if r["campaign"] == 1 and r["cycle"] == 2]
    assert len(first) == len(second) == 3
    assert all(r["reason"] == "random_neighbor" for r in second)
    assert all(r["mutation_parent_index"] == max(r["index"] for r in first) for r in second)
    assert all(r["filter_attempts"] == 6 for r in first + second)


def test_collision_ends_campaign_and_reinsertion_matches_author_frequency_rule():
    scenes = sample("cut_in", 12, "development")

    def execute(scene):
        return PaidExecution(
            QueryFeedback(scene.scenario_id, True, SafetyMeasurements(0, 0), 1), []
        )

    result = run_isolated(
        run,
        {
            "candidates": scenes,
            "classifier": ConstantRisk(),
            "initial_indices": [4, 2],
            "budget": 8,
            "config": Config(mutants=8),
        },
        QueryLedger(scenes, 8, execute),
    )
    assert len(result["campaigns"]) == 6
    assert all(row["cycle"] in (0, 1) for row in result["queries"])
    for row in result["campaigns"]:
        assert row["reinsert_probability"] == pytest.approx(
            1 - row["seed_selections"] / row["campaign"]
        )
