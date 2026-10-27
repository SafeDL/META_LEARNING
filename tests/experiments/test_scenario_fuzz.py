from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.scenarios import sample
from testing_methods.baselines.scenario_fuzz.graphs import graph
from testing_methods.baselines.scenario_fuzz.model import (
    Config,
    ScenarioRiskNetwork,
    batch_graphs,
    focal_loss,
    load,
    train,
)
from testing_methods.common.response import ResponseArchive


def test_graph_uses_public_waypoints_and_distinguishes_planned_interactions():
    scene = sample("cut_in", 1, "history")[0]
    first = graph(scene)
    mirrored = graph(
        replace(
            scene, categories={"side": "right" if scene.categories["side"] == "left" else "left"}
        )
    )
    assert first.x.shape[1] == 8
    assert first.edge_attr.shape[1] == 3
    assert first.weather_attr.shape == (1, 8)
    assert torch.equal(first.x, graph(replace(scene, scenario_id="another_system_name")).x)
    assert not torch.equal(first.x, mirrored.x)
    delayed = graph(
        replace(
            scene,
            parameters={
                **scene.parameters,
                "event_time_s": scene.parameters["event_time_s"] + 0.01,
            },
        )
    )
    assert not torch.equal(first.x, delayed.x)
    assert first.edge_index.max() < len(first.x)
    assert first.edge_graph_index.max() < len(first.edge_attr)
    source, destination = first.edge_index
    assert first.edge_attr[:, 0] == pytest.approx(
        torch.linalg.vector_norm(first.x[destination, :3] - first.x[source, :3], dim=-1).numpy()
    )


def test_dual_graph_prediction_does_not_depend_on_other_scenarios_in_the_batch():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    graphs = [graph(sample(kind, 1, "history")[0]) for kind in ("cut_in", "cut_out")]
    model = ScenarioRiskNetwork(hidden_dim=8, heads=2).eval()
    model.fit_normalization(graphs)
    with torch.no_grad():
        individual = torch.cat([model(batch_graphs([data])) for data in graphs])
        combined = model(batch_graphs(graphs))
        reversed_batch = model(batch_graphs(list(reversed(graphs)))).flip(0)
    assert combined.numpy() == pytest.approx(individual.numpy(), abs=1e-6)
    assert reversed_batch.numpy() == pytest.approx(individual.numpy(), abs=1e-6)
    assert torch.isfinite(combined).all()


def test_focal_loss_uses_logits_and_gives_both_classes_the_correct_gradient():
    logits = torch.tensor([-4.0, 4.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0])
    good = focal_loss(logits, labels)
    bad = focal_loss(-logits, labels)
    assert good < bad / 1000
    good.backward()
    assert logits.grad[0] > 0 and logits.grad[1] < 0


def test_graph_risk_builds_from_history_and_saves_finite_model_with_constant_weather(tmp_path):
    scenes = sample("cut_in", 16, "history")
    archive = ResponseArchive(
        "history", "rule", "idm", "history", scenes, np.arange(16) % 4 == 0, np.ones((16, 2))
    )
    path = tmp_path / "graph.pt"
    report = train([archive], path, config=Config(hidden_dim=4, heads=1, epochs=1, batch_size=8))
    model, metadata = load(path)
    assert report["training_rows"] + report["validation_rows"] == 16
    assert metadata["history_ids"] == ["history"]
    with torch.no_grad():
        assert torch.isfinite(model(batch_graphs([graph(scenes[0])]))).all()
    with pytest.raises(ValueError, match="historical objects only"):
        train([replace(archive, role="development")], tmp_path / "leak.pt")
    with pytest.raises(ValueError, match="Held family"):
        train([archive], tmp_path / "held.pt", held_family="rule")
