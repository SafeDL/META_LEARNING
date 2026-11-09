from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from experiments import pilot
from experiments.domain import Domain
from experiments.executions import ExecutionBank
from experiments.history import completed_archives, finalize_task
from experiments.joint_validation import Config, run_one
from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.rss import measurement_contract
from experiments.schema import QueryFeedback, SafetyMeasurements, Scenario
from experiments.storage import write_json
from testing_methods.baselines.behavexplor.pool import mutate as behavior_mutate
from testing_methods.baselines.samota.pool import mixed_distance
from testing_methods.baselines.scenario_fuzz import model as fuzz_model
from testing_methods.baselines.scenario_fuzz.graphs import ScenarioGraph
from testing_methods.baselines.scenario_fuzz.pool import mutate as fuzz_mutate
from testing_methods.baselines.scenario_fuzz.pool import run as fuzz_run
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import Config as TrainingConfig
from testing_methods.history_transfer.training import load_model, train

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


class ConstantRisk(torch.nn.Module):
    def __init__(self, domain):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(-1.0))
        self.graph_specification = {
            "parameter_definition": domain.to_dict(),
            "graph_definition": GRAPH_DEFINITION,
        }

    def forward(self, data):
        return self.logit.expand(len(data.weather_attr))


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


@pytest.mark.parametrize("method", pilot.PRIMARY_BASELINES)
def test_all_six_baselines_share_the_registered_bank_and_fixed_paid_budget(
    tmp_path, monkeypatch, method
):
    bank = EngineeringBank()
    config = Config(candidate_count=10, initial_count=4, budget=7)
    if method != "random":
        module = pilot.BASELINES[method]
        changes = {
            "samota": {"rbf_epochs": 2},
            "av_bo_rf_ei": {"n_estimators": 10, "max_depth": 3},
            "paired_system_active_evaluation": {
                "training_epochs": 3,
                "mc_draws": 4,
                "batch_size": 2,
                "shortlist_size": 4,
                "local_samples": 8,
                "paired_samples": 3,
            },
            "scenario_fuzz": {"mutants": 8},
        }
        monkeypatch.setattr(module, "DEFAULT", replace(module.DEFAULT, **changes.get(method, {})))
    options = {}
    classifiers = {}
    if method == "scenario_fuzz":
        classifiers[method] = ConstantRisk(bank.domain)
        options = {"graph_fn": parameter_graph_fixture, "graph_definition": GRAPH_DEFINITION}
    saved, ledger = pilot.baseline_task(
        tmp_path, bank, history(bank), classifiers, config, 11, {"method": method}, **options
    )
    assert saved["specification"]["parameter_definition"] == bank.domain.to_dict()
    assert saved["metrics"]["queries"] == 7
    assert saved["metrics"]["collision_recall"] is None
    assert len(set(saved["result"]["selected_indices"])) == 7
    assert [r["purpose"] for r in ledger.records] == ["initial"] * 4 + ["target"] * 3
    assert ledger.domain.to_dict() == bank.domain.to_dict()
    with pytest.raises(ValueError, match="fixed candidate library"):
        ledger.register(replace(bank.candidates[0], scenario_id="extra"))


@pytest.mark.parametrize("control", ["joint", "collision", "fixed_weights", "no_history"])
def test_joint_and_three_controls_use_the_registered_task_dispatch(tmp_path, control):
    bank = EngineeringBank()
    archives = history(bank)
    model_path = tmp_path / "model"
    train(
        archives,
        model_path,
        config=TrainingConfig(steps=1, initial_count=2, budget=5, historical_context=4),
        kernel_type=SharedPrivateKernel,
    )
    models = tuple(
        load_model(model_path / f"{part}.pt", kernel_type=SharedPrivateKernel)[0]
        for part in ("target_kernel", "historical_encoder")
    )
    options = {
        "feedback_mode": "collision" if control == "collision" else "joint",
        "update_source_weights": control != "fixed_weights",
    }
    if control == "no_history":
        models, archives = (None, None), []
        options["raw_kernel_type"] = SharedPrivateKernel
    saved, ledger = run_one(
        tmp_path / "run",
        bank,
        models,
        archives,
        Config(candidate_count=10, initial_count=4, budget=7),
        11,
        **options,
    )
    assert saved["result"]["response_kernel"] == "SharedPrivateKernel"
    assert saved["specification"]["parameter_definition"] == bank.domain.to_dict()
    assert len(ledger.records) == len(set(saved["result"]["selected_indices"])) == 7
    if control == "no_history":
        assert saved["result"]["component_names"] == ["raw_target"]
    archived = finalize_task(
        tmp_path / "completed.json.gz",
        ledger,
        task_id="task",
        sut_id="fixture",
        sut_version="engineering",
        family="rule",
        lineage="fixture",
        environment_seed=81001,
        completed_order=1,
        branch_id="fixture",
    )
    restored = completed_archives([archived], before_order=2, branch_id="fixture")[0]
    assert restored.domain.to_dict() == bank.domain.to_dict()


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


def test_dispatch_rejects_another_bank_definition_and_a_budget_larger_than_the_library(tmp_path):
    bank = EngineeringBank()
    model = SharedPrivateKernel(2).double()
    model.construction_specification = {
        "history_fingerprints": {},
        "scenario": "front_brake",
        "parameter_definition": replace(
            bank.domain, fixed_parameters={"ego_speed_mps": 12}
        ).to_dict(),
    }
    config = Config(candidate_count=10, initial_count=2, budget=3)
    with pytest.raises(ValueError, match="different parameter definitions"):
        run_one(tmp_path / "wrong", bank, (model, None), [], config, 11)
    with pytest.raises(ValueError, match="budget exceeds"):
        pilot.baseline_task(
            tmp_path / "budget", bank, [], {}, replace(config, budget=11), 11, {"method": "random"}
        )


@pytest.mark.parametrize("dimension", [2, 7])
def test_mutation_and_local_distances_use_all_and_only_registered_free_dimensions(dimension):
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
    np.testing.assert_array_equal(
        mixed_distance(definition.features(scenes), "front_brake", domain=definition),
        [[0, 1], [1, 0]],
    )
    generator = np.random.default_rng(11)
    for mutate in (fuzz_mutate, behavior_mutate):
        for local in (False, True):
            scene = mutate(scenes[0], generator, local=local, domain=definition)
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
    classifiers = pilot.historical_classifiers(history(bank), tmp_path, **options)
    assert (
        classifiers["scenario_fuzz"].graph_specification["parameter_definition"]
        == bank.domain.to_dict()
    )
    resumed = pilot.historical_classifiers(history(bank), tmp_path, train_missing=False, **options)
    assert (
        resumed["scenario_fuzz"].graph_specification
        == classifiers["scenario_fuzz"].graph_specification
    )
    with pytest.raises(ValueError, match="another public graph definition"):
        pilot.historical_classifiers(
            history(bank),
            tmp_path,
            train_missing=False,
            **{**options, "graph_definition": {"other": True}},
        )
