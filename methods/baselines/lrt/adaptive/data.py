"""Frozen-policy historical trajectories and labels for their future prefixes."""

import numpy as np
import torch

from methods.baselines.dart.train import HistoricalOracle
from methods.baselines.hgt.config import group_of
from methods.baselines.hgt.io import write_json
from benchmark.timing import now

from ..data import historical_case, task_settings
from ..selection import select
from ..train import load_value
from .config import ROOT, SEED, horizons
from .model import RemainingDiscoveries


class FeatureRecorder:
    def __init__(self, network):
        self.network = network
        self.records = []

    def evaluate(self, models, remaining, anchor, queries, evidence, budget):
        lengths = horizons(budget - len(queries))
        self.records.append(
            {
                "step": len(queries),
                "found": sum(q["risk"] > 0.5 for q in queries),
                "lengths": lengths,
                "features": self.network.features(
                    models, remaining, anchor, queries, evidence, budget, lengths
                ).cpu(),
            }
        )
        return {"query": len(queries), "stop": False}


def generate(workspace):
    folder = ROOT / workspace.root.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    value = load_value(SEED, device=device, root=workspace.root)
    network = RemainingDiscoveries(value).to(device).eval()
    for split in ("train", "validation"):
        budget, initial, _ = task_settings(split)
        for name in workspace.history:
            for condition in workspace.archives:
                path = folder / "historical" / split / f"{name}_{condition}.pt"
                if path.exists():
                    continue
                started = now()
                common, prior, truth = historical_case(
                    name, condition, SEED, split, workspace=workspace
                )
                recorder = FeatureRecorder(network)
                trace = select(
                    **common,
                    transfer_prior=prior,
                    value_model=value,
                    oracle=HistoricalOracle(truth),
                    budget=budget,
                    initial_queries=initial,
                    stopper=recorder,
                )
                cumulative = np.r_[
                    0, np.cumsum([q["risk"] > 0.5 for q in trace["queries"]])
                ]
                for record in recorder.records:
                    t = record["step"]
                    record["returns"] = [
                        int(cumulative[t + b] - cumulative[t])
                        for b in record["lengths"]
                    ]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "source": name,
                        "group": group_of(name),
                        "condition": condition,
                        "split": split,
                        "budget": budget,
                        "initial": initial,
                        "positives": int((truth > 0.5).sum()),
                        "found": cumulative[1:].tolist(),
                        "records": recorder.records,
                    },
                    path,
                )
                write_json(
                    path.with_suffix(".json"),
                    {
                        "source": name,
                        "group": group_of(name),
                        "condition": condition,
                        "split": split,
                        "budget": budget,
                        "initial": initial,
                        "queries": trace["queries"],
                        "selected_indices": trace["selected_indices"],
                        "elapsed_s": now() - started,
                        "physical_calls": 0,
                    },
                )
                print(
                    f"HISTORY {workspace.root.name} {split} {name} {condition}",
                    flush=True,
                )


def load_tasks(family, split):
    return [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted((ROOT / family / "historical" / split).glob("*.pt"))
    ]
