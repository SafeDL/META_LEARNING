"""Retrospective replay of completed paid tasks before and after a shared-model update."""

import copy
from pathlib import Path

import numpy as np

from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.selection import select

from .history import build_source, model_version
from .rss import measurement_contract
from .schema import QueryFeedback, SafetyMeasurements
from .storage import read_json, write_json


class RecordedResponses:
    def __init__(self, archive):
        self.archive = archive
        self.next_index = 0

    def query(self, index, *, purpose):
        if index != self.next_index:
            raise ValueError("Completed-task diagnostics must retain the executed query order")
        self.next_index += 1
        archive = self.archive
        return QueryFeedback(
            archive.scenarios[index].scenario_id,
            bool(archive.collision[index]),
            SafetyMeasurements(
                *archive.safety[index], rss_margin=float(archive.continuous_margin[index])
            ),
            0,
            measurement_contract(),
        )


def replay(models, sources, archive, config):
    count = len(archive.scenarios)
    initial = min(config.initial_count, max(0, count - 1))
    priors = build_source(copy.deepcopy(models[1]).to(config.device), sources, archive.scenarios)
    result = select(
        archive.scenarios,
        RecordedResponses(archive),
        pretrained=models[0],
        sources=priors,
        initial_indices=list(range(count)),
        budget=count,
        device=config.device,
        batch_size=config.batch_size,
        snapshot_counts=(),
    )
    probability = np.array([query["prequery_probability"] for query in result["queries"]])
    p, y = (
        np.clip(probability[initial:], 1e-12, 1 - 1e-12),
        archive.collision[initial:].astype(float),
    )
    return {
        "initial_count": initial,
        "evaluated_records": len(y),
        "prequery_probabilities": probability.tolist(),
        "collision_nll": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        "brier": float(np.mean((p - y) ** 2)),
        "resources": result["resources"],
        "shared_training_contains_task": any(
            archive.source_id
            in getattr(model, "construction_specification", {}).get("history_fingerprints", {})
            for model in models
        ),
    }


def compare(destination, before, after, history, completed, config):
    path = Path(destination)
    specification = {
        "before": [model_version(model) for model in before],
        "after": [model_version(model) for model in after],
        "history": archive_fingerprints(history),
        "tasks": archive_fingerprints(completed),
        "initial_count": config.initial_count,
        "batch_size": config.batch_size,
        "device": config.device,
    }
    if path.exists():
        saved = read_json(path)
        if saved["specification"] != specification:
            raise ValueError("Completed-task replay differs from its model/data snapshot")
        return saved
    rows = []
    for archive in completed:
        sources = [source for source in history if source.source_id != archive.source_id]
        left, right = [replay(models, sources, archive, config) for models in (before, after)]
        rows.append(
            {
                "source_id": archive.source_id,
                "before": left,
                "after": right,
                "nll_change": right["collision_nll"] - left["collision_nll"],
                "brier_change": right["brier"] - left["brier"],
                "record_ids": archive.provenance["record_ids"],
            }
        )
    result = {
        "specification": specification,
        "tasks": rows,
        "interpretation": "retrospective replay of completed records, including shared-training records; not independent validation",
        "new_physical_calls": 0,
        "new_paid_queries": 0,
        "publication_gate": False,
    }
    write_json(path, result)
    return result
