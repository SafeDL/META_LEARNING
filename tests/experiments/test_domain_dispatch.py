from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from experiments import baselines
from experiments.domain import Domain
from experiments.executions import ExecutionBank
from experiments.history import finalize_task
from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.rss import measurement_contract
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from experiments.storage import write_json
from testing_methods.baselines.scenario_fuzz import model as fuzz_model
from testing_methods.baselines.scenario_fuzz.graphs import ScenarioGraph
from testing_methods.baselines.scenario_fuzz.pool import mutate as fuzz_mutate
from testing_methods.baselines.scenario_fuzz.pool import run as fuzz_run
from testing_methods.common.response import ResponseArchive

GRAPH_DEFINITION = {"engineering_fixture": True, "input": "public parameters only"}


def parameter_graph_fixture(scene):
    return ScenarioGraph(
        x=torch.tensor(
            [
                [0, 0, 0, 1, 0, 1, 0, 0.3],
                [scene.parameters["gap_m"], 0, 0, 3, 0, 1, 0, 0.3],
                [40, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([[1.0, 0, 0]] * 4),
        edge_graph_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        weather_attr=torch.zeros(1, 8),
    )


class EngineeringBank:
    """Analytic safe traces test interfaces; they are not ADS evaluation data."""

    def __init__(self):
        self.domain = Domain(
            {"gap_m": (12, 40), "deceleration_mps2": (2, 6)},
            {"gap_m": 1, "deceleration_mps2": 0.1},
            {},
            {"ego_speed_mps": 50 / 3.6, "event_time_s": 1},
        )
        self.candidates = [
            Scenario(str(i), "front_brake", {"gap_m": 12 + 3 * i, "deceleration_mps2": 2 + 0.4 * i})
            for i in range(10)
        ]
        self.manifest = {
            "contract": {"engineering_fixture": True},
            "environment_seed": 81001,
            "parameter_definition": self.domain.to_dict(),
        }

    def __call__(self, scene):
        v = self.domain.fixed_parameters["ego_speed_mps"]
        frames = []
        for t in np.arange(41) / 20:
            frames.append(
                {
                    "time_s": float(t),
                    "collision": False,
                    "control": {"steering": 0, "acceleration": 0},
                    "actors": [
                        {
                            "actor_id": name,
                            "x": v * t + offset,
                            "y": 0,
                            "speed": v,
                            "heading": 0,
                            "length": 5,
                            "width": 2,
                        }
                        for name, offset in [("ego", 0), ("lead", scene.parameters["gap_m"] + 5)]
                    ],
                }
            )
        return PaidExecution(
            QueryFeedback(
                scene.scenario_id,
                False,
                SafetyMeasurements(scene.parameters["gap_m"], float("inf")),
                0,
            ),
            frames,
        )


def history(bank):
    return [
        ResponseArchive(
            name,
            "rule",
            name,
            "history",
            bank.candidates,
            np.arange(10) % 3 == 0,
            np.ones((10, 2)),
            rss_margin=np.linspace(-0.6, 0.3, 10),
            rss_contract=measurement_contract(),
            domain=bank.domain,
        )
        for name in ("a", "b")
    ]


def test_execution_bank_reads_registered_definition_and_rejects_wrong_coordinates(tmp_path):
    bank = EngineeringBank()
    record = {
        **bank.manifest,
        "format": "collision-execution-bank",
        "status": "complete",
        "scenarios": [asdict(s) for s in bank.candidates],
    }
    write_json(tmp_path / "manifest.json", record)
    restored = ExecutionBank(tmp_path)
    assert restored.domain.to_dict() == bank.domain.to_dict()
    record["scenarios"][0]["parameters"]["gap_m"] = 0
    write_json(tmp_path / "manifest.json", record)
    with pytest.raises(ValueError, match="registered range"):
        ExecutionBank(tmp_path)


def test_query_registration_and_archives_cannot_change_the_paid_parameter_definition(tmp_path):
    bank = EngineeringBank()
    ledger = QueryLedger(bank.candidates[:1], 2, bank, domain=bank.domain)
    assert ledger.register(bank.candidates[1]) == 1
    assert not ledger.records
    with pytest.raises(ValueError, match="registered range"):
        ledger.register(
            replace(bank.candidates[2], parameters={"gap_m": 0, "deceleration_mps2": 2})
        )
    ledger.query(0)
    other = replace(bank.domain, fixed_parameters={"ego_speed_mps": 12})
    with pytest.raises(ValueError, match="different parameter definitions"):
        finalize_task(
            tmp_path / "wrong.json.gz",
            ledger,
            task_id="task",
            sut_id="fixture",
            sut_version="fixture",
            family="rule",
            lineage="fixture",
            environment_seed=1,
            completed_order=1,
            branch_id="fixture",
            domain=other,
        )


def test_new_graph_training_loading_and_query_require_the_same_registered_builder(tmp_path):
    bank = EngineeringBank()
    with pytest.raises(ValueError, match="registered public graph builder"):
        fuzz_model.train(history(bank), tmp_path / "wrong.pt", config=fuzz_model.Config(epochs=1))
    path = tmp_path / "graph.pt"
    fuzz_model.train(
        history(bank),
        path,
        config=fuzz_model.Config(epochs=1, hidden_dim=4, heads=1, batch_size=4),
        graph_fn=parameter_graph_fixture,
        graph_definition=GRAPH_DEFINITION,
    )
    model, checkpoint = fuzz_model.load(path)
    assert checkpoint["parameter_definition"] == bank.domain.to_dict()
    assert model.graph_specification["graph_definition"] == GRAPH_DEFINITION
    with pytest.raises(ValueError, match="definitions differ"):
        fuzz_run(
            bank.candidates,
            None,
            classifier=model,
            initial_indices=[0],
            budget=2,
            domain=bank.domain,
            graph_fn=parameter_graph_fixture,
            graph_definition={"engineering_fixture": "different"},
        )
    with pytest.raises(ValueError, match="registered public graph builder"):
        fuzz_run(
            bank.candidates,
            None,
            classifier=model,
            initial_indices=[0],
            budget=2,
            domain=bank.domain,
        )


@pytest.mark.parametrize("dimension", [2, 7])
def test_mutation_uses_all_and_only_registered_free_dimensions(dimension):
    definition = Domain(
        {f"p{i}": (0, 1) for i in range(dimension)},
        {f"p{i}": 0.1 for i in range(dimension)},
        {},
        {"speed": 10},
    )
    scenes = [
        Scenario(str(i), "front_brake", {name: float(i) for name in definition.bounds})
        for i in range(2)
    ]
    generator = np.random.default_rng(11)
    for local in (False, True):
        scene = fuzz_mutate(scenes[0], generator, local=local, domain=definition)
        definition.validate(scene)
        assert len(scene.parameters) == dimension and "speed" not in scene.parameters


def attempt_registration(oracle, *, proposed):
    oracle.query(0, purpose="initial")
    oracle.register(proposed)
    raise AssertionError("A fixed library accepted an additional coordinate")


def test_fixed_library_rejects_registration_from_the_isolated_selector():
    bank = EngineeringBank()
    ledger = QueryLedger(bank.candidates[:1], 2, bank, domain=bank.domain, allow_registration=False)
    with pytest.raises(ValueError, match="fixed candidate library"):
        run_isolated(
            attempt_registration,
            {"proposed": bank.candidates[1]},
            ledger,
        )
    assert len(ledger.candidates) == len(ledger.records) == 1
    assert ledger.records[0]["status"] == "complete"


def test_graph_classifier_dispatch_reuses_only_the_same_registered_definition(
    tmp_path, monkeypatch
):
    bank = EngineeringBank()
    monkeypatch.setattr(
        fuzz_model, "DEFAULT", fuzz_model.Config(epochs=1, hidden_dim=4, heads=1, batch_size=4)
    )
    options = {
        "names": ("scenario_fuzz",),
        "graph_fn": parameter_graph_fixture,
        "graph_definition": GRAPH_DEFINITION,
    }
    classifiers = baselines.historical_classifiers(history(bank), tmp_path, **options)
    assert (
        classifiers["scenario_fuzz"].graph_specification["parameter_definition"]
        == bank.domain.to_dict()
    )
    resumed = baselines.historical_classifiers(
        history(bank), tmp_path, train_missing=False, **options
    )
    assert (
        resumed["scenario_fuzz"].graph_specification
        == classifiers["scenario_fuzz"].graph_specification
    )
    with pytest.raises(ValueError, match="another public graph definition"):
        baselines.historical_classifiers(
            history(bank),
            tmp_path,
            train_missing=False,
            **{**options, "graph_definition": {"other": True}},
        )
