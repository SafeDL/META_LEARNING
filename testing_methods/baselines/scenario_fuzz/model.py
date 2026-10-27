"""ScenarioFuzz's dual GAT streams with explicit edge-graph batching and focal loss.

Architecture follows AtongWang/ScenarioFuzz commit
49360794846786b133799a62827528bc4644bc62, src/scenario_eval_model/eval_model.py.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool

from testing_methods.common.response import archive_fingerprints

from .graphs import graph


@dataclass(frozen=True)
class Config:
    hidden_dim: int = 64
    heads: int = 4
    epochs: int = 1000
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 1409


DEFAULT = Config()


def batch_graphs(graphs):
    return Batch.from_data_list(graphs, follow_batch=["edge_attr"])


def focal_loss(logits, labels):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    alpha = torch.where(labels > 0, 0.95, 0.05)
    return (alpha * (1 - torch.exp(-bce)) ** 2 * bce).mean()


class ScenarioRiskNetwork(nn.Module):
    def __init__(self, *, hidden_dim=64, heads=4):
        super().__init__()
        self.hidden_dim, self.num_heads = hidden_dim, heads
        self.node_gat1 = GATConv(8, hidden_dim, heads=heads)
        self.node_gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.edge_gat1 = GATConv(3, hidden_dim, heads=heads)
        self.edge_gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.weather_mlp1 = nn.Linear(8, hidden_dim)
        self.weather_mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.node_bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.node_bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.edge_bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.edge_bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.weather_bn1 = nn.BatchNorm1d(hidden_dim)
        self.readout_mlp1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.readout_mlp2 = nn.Linear(hidden_dim, 1)
        self.register_buffer("position_mean", torch.zeros(3))
        self.register_buffer("position_std", torch.ones(3))
        self.register_buffer("distance_min", torch.tensor(0.0))
        self.register_buffer("distance_range", torch.tensor(1.0))

    def fit_normalization(self, graphs):
        positions = torch.cat([data.x[:, :3] for data in graphs])
        distances = torch.cat([data.edge_attr[:, 0] for data in graphs])
        self.position_mean.copy_(positions.mean(0))
        std = positions.std(0)
        self.position_std.copy_(torch.where(std > 0, std, torch.ones_like(std)))
        self.distance_min.copy_(distances.min())
        span = distances.max() - distances.min()
        self.distance_range.copy_(span if span > 0 else torch.ones_like(span))

    def forward(self, data):
        x, edge_attr = data.x.clone(), data.edge_attr.clone()
        x[:, :3] = (x[:, :3] - self.position_mean) / self.position_std
        edge_attr[:, 0] = (edge_attr[:, 0] - self.distance_min) / self.distance_range
        node_h = F.relu(self.node_gat1(x, data.edge_index))
        node_h = self.node_bn1(node_h)
        node_h = F.dropout(node_h, p=0.5, training=self.training)
        node_h = self.node_bn2(self.node_gat2(node_h, data.edge_index))
        node_h = node_h.view(-1, self.hidden_dim, self.num_heads).mean(-1)
        edge_h = F.relu(self.edge_gat1(edge_attr, data.edge_graph_index))
        edge_h = self.edge_bn1(edge_h)
        edge_h = F.dropout(edge_h, p=0.5, training=self.training)
        edge_h = self.edge_bn2(self.edge_gat2(edge_h, data.edge_graph_index))
        edge_h = edge_h.view(-1, self.hidden_dim, self.num_heads).mean(-1)
        weather_h = self.weather_bn1(F.relu(self.weather_mlp1(data.weather_attr)))
        weather_h = self.weather_mlp2(weather_h)
        combined = torch.cat(
            [
                global_mean_pool(node_h, data.batch),
                global_mean_pool(edge_h, data.edge_attr_batch),
                weather_h,
            ],
            dim=-1,
        )
        return self.readout_mlp2(F.relu(self.readout_mlp1(combined))).squeeze(-1)


def train(history, destination, *, held_family=None, config=DEFAULT):
    history = tuple(history)
    if not history or any(a.role != "history" for a in history):
        raise ValueError("ScenarioFuzz's graph risk model uses allowed historical objects only")
    if held_family is not None and any(a.family == held_family for a in history):
        raise ValueError("Held family entered ScenarioFuzz graph-model training")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    scenarios = [scene for archive in history for scene in archive.scenarios]
    x = np.concatenate([a.x for a in history])
    unique, representative, groups = np.unique(x, axis=0, return_index=True, return_inverse=True)
    graphs = [graph(scenarios[i]) for i in representative]
    labels = torch.tensor(np.concatenate([a.collision for a in history]), dtype=torch.float32)
    order = generator.permutation(len(unique))
    validation_groups = set(order[: max(1, round(0.2 * len(unique)))].tolist())
    validation = np.asarray([i for i, group in enumerate(groups) if group in validation_groups])
    training = np.asarray([i for i, group in enumerate(groups) if group not in validation_groups])
    model = ScenarioRiskNetwork(hidden_dim=config.hidden_dim, heads=config.heads)
    model.fit_normalization([data for i, data in enumerate(graphs) if i not in validation_groups])
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best, state, curve = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        indices = generator.permutation(training)
        for batch in np.array_split(indices, int(np.ceil(len(indices) / config.batch_size))):
            data = batch_graphs([graphs[groups[i]] for i in batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(data), labels[batch].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for batch in np.array_split(
                validation, int(np.ceil(len(validation) / config.batch_size))
            ):
                data = batch_graphs([graphs[groups[i]] for i in batch]).to(device)
                loss_sum += float(focal_loss(model(data), labels[batch].to(device))) * len(batch)
        score = loss_sum / len(validation)
        curve.append({"epoch": epoch, "historical_validation_focal_loss": score})
        if score < best:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            print(
                "ScenarioFuzz",
                scenarios[0].kind,
                "epoch",
                epoch,
                "/",
                config.epochs,
                "validation_loss",
                round(score, 6),
                "wall_s",
                round(time.perf_counter() - started),
                flush=True,
            )
    checkpoint = {
        "format": "scenariofuzz-highway-graph",
        "state_dict": state,
        "config": asdict(config),
        "history_ids": [a.source_id for a in history],
        "history_fingerprints": archive_fingerprints(history),
        "held_family": held_family,
        "curve": curve,
        "training_s": time.perf_counter() - started,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_device": str(device),
        "training_threads": 1,
        "graph_scope": "public straight-road waypoints and scheduled background paths",
        "author_code_corrections": [
            "edge-stream line-graph adjacency and batching",
            "focal loss receives pre-sigmoid logits",
            "constant position/weather features stay finite",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return {k: v for k, v in checkpoint.items() if k != "state_dict"}


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["format"] != "scenariofuzz-highway-graph":
        raise ValueError("ScenarioFuzz requires its task-specific historical graph model")
    config = checkpoint["config"]
    model = ScenarioRiskNetwork(hidden_dim=config["hidden_dim"], heads=config["heads"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False), checkpoint
