"""Pinned ScenarioFuzz architecture comparison and edge-batching diagnosis."""

import ast
import urllib.request

import torch
import torch_geometric.nn as pyg_nn
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool

from testing_methods.baselines.scenario_fuzz.graphs import ScenarioGraph, graph
from testing_methods.baselines.scenario_fuzz.model import ScenarioRiskNetwork, batch_graphs

from .discovery_config import write_json
from .protocol import ROOT
from .scenarios import sample

COMMIT = "49360794846786b133799a62827528bc4644bc62"
SOURCE = (
    "https://raw.githubusercontent.com/AtongWang/ScenarioFuzz/"
    f"{COMMIT}/src/scenario_eval_model/eval_model.py"
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    source = urllib.request.urlopen(SOURCE, timeout=30).read().decode("utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImprovedGNNBinaryClassificationModel"
    )
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "pyg_nn": pyg_nn,
        "global_mean_pool": global_mean_pool,
    }
    exec(compile(module, SOURCE, "exec"), namespace)
    reference = namespace[cls.name](8, 3, 8, 8, num_heads=2).eval()
    model = ScenarioRiskNetwork(hidden_dim=8, heads=2).eval()
    reference.load_state_dict({k: model.state_dict()[k] for k in reference.state_dict()})
    cycle = torch.tensor([[0, 1, 2], [1, 2, 0]])
    data = batch_graphs(
        [
            ScenarioGraph(
                x=torch.randn(3, 8),
                edge_index=cycle,
                edge_attr=torch.randn(3, 3),
                edge_graph_index=cycle,
                weather_attr=torch.zeros(1, 8),
            )
        ]
    )
    with torch.no_grad():
        expected = reference(
            data.x,
            data.edge_attr,
            data.weather_attr,
            data.edge_index,
            data.batch,
            data.edge_attr_batch,
        ).squeeze(-1)
        error = float((model(data).sigmoid() - expected).abs().max())
    if error > 1e-7:
        raise ValueError("ScenarioFuzz architecture differs on matching node/edge topology")
    graphs = [graph(sample(kind, 1, "history")[0]) for kind in ("cut_in", "cut_out")]
    model.fit_normalization(graphs)

    def legacy_prediction(graphs):
        batch = batch_graphs(graphs)
        batch.x[:, :3] = (batch.x[:, :3] - model.position_mean) / model.position_std
        batch.edge_attr[:, 0] = (batch.edge_attr[:, 0] - model.distance_min) / model.distance_range
        with torch.no_grad():
            return reference(
                batch.x,
                batch.edge_attr,
                batch.weather_attr,
                batch.edge_index,
                batch.batch,
                batch.edge_attr_batch,
            ).squeeze(-1)

    original_order = legacy_prediction(graphs)
    changed_order = legacy_prediction(list(reversed(graphs))).flip(0)
    original_batch_error = float((original_order - changed_order).abs().max())
    with torch.no_grad():
        corrected_order = model(batch_graphs(graphs)).sigmoid()
        corrected_reverse = model(batch_graphs(list(reversed(graphs)))).sigmoid().flip(0)
    corrected_batch_error = float((corrected_order - corrected_reverse).abs().max())
    if corrected_batch_error > 1e-6:
        raise ValueError("ScenarioFuzz line-graph adapter mixes separate scenario graphs")
    report = {
        "status": "passed",
        "author_source": SOURCE,
        "author_commit": COMMIT,
        "same_input_same_weight_probability_error": error,
        "author_batch_permutation_probability_error": original_batch_error,
        "corrected_batch_permutation_probability_error": corrected_batch_error,
        "corrections": [
            "edge features use line-graph adjacency with edge-count batch offsets",
            "focal loss receives logits; inference applies sigmoid exactly once",
            "constant planar/weather features do not divide by zero",
        ],
        "task_adaptations": [
            "fixed highway road and scheduled public background waypoint paths",
            "node features extend the original six with planned time and speed",
        ],
        "new_physical_calls": 0,
    }
    write_json(
        ROOT
        / "experiment_outputs/benchmark/collision/discovery/domain_comparison/scenario_fuzz_author_audit.json",
        report,
    )
    print(report, flush=True)
    return report


if __name__ == "__main__":
    main()
