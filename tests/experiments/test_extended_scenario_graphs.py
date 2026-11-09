import torch

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
    assert predictions.shape == (len(DOMAINS) + 1,) and torch.isfinite(predictions).all()
