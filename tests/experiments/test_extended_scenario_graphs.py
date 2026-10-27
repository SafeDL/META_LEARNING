import zipfile

import pytest
import torch

from experiments.protocol import ROOT
from experiments.scenarios import DOMAINS, sample
from testing_methods.baselines.scenario_fuzz.graphs import graph
from testing_methods.baselines.scenario_fuzz.model import ScenarioRiskNetwork, batch_graphs


def test_extended_graphs_use_public_paths_and_support_batching():
    graphs = [graph(sample(kind, 2, "history")[0]) for kind in DOMAINS]
    opened, restricted = [graph(s) for s in sample("cut_in_interaction", 2, "history")]
    assert restricted.x.shape[0] == opened.x.shape[0] + 4
    for data in graphs + [restricted]:
        assert data.x.shape[1] == 8 and data.edge_attr.shape[1] == 3
        assert torch.isfinite(data.x).all() and torch.isfinite(data.edge_attr).all()
        assert data.edge_graph_index.max() < data.edge_index.shape[1]
    model = ScenarioRiskNetwork().eval()
    with torch.no_grad():
        predictions = model(batch_graphs(graphs + [restricted]))
    assert predictions.shape == (6,) and torch.isfinite(predictions).all()


def test_original_two_graph_domains_match_frozen_source_exactly():
    path = (
        ROOT
        / "experiment_outputs/benchmark/collision/discovery/domain_comparison/source_snapshot.zip"
    )
    if not path.exists():
        pytest.skip("The local Stage B/C source replay artifact is unavailable")
    namespace = {"__name__": "frozen_scenario_graph"}
    with zipfile.ZipFile(path) as stream:
        source = stream.read("methods/baselines/scenario_fuzz/graphs.py").decode("utf-8")
        source = source.replace("from benchmark.", "from experiments.")
        exec(compile(source, str(path), "exec"), namespace)
    for kind in ("cut_in", "cut_out"):
        for scenario in sample(kind, 16, "history"):
            old, new = namespace["graph"](scenario), graph(scenario)
            for key in ("x", "edge_index", "edge_attr", "edge_graph_index", "weather_attr"):
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
