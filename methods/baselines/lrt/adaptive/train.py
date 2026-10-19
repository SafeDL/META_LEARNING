"""Choose a count head and threshold using historical complete trajectories."""

import numpy as np
import torch

from methods.baselines.hgt.io import read_json, write_json

from methods.common.counts import count_cdf, count_log_prob, expected_count
from ..train import load_value
from .config import BATCH_SIZE, INTERVAL, RETENTION, ROOT, SEED, STEPS, THRESHOLDS
from .data import load_tasks
from .model import RemainingDiscoveries, training_loss


def balanced_mean(rows, key):
    return float(
        np.mean(
            [
                np.mean([r[key] for r in rows if r["group"] == group])
                for group in sorted({r["group"] for r in rows})
            ]
        )
    )


@torch.no_grad()
def predictions(network, task):
    device = next(network.parameters()).device
    result = []
    for record in task["records"]:
        parameters = network(record["features"][:, -1].to(device))
        length = record["lengths"][-1]
        probability = count_cdf(parameters, length, record["found"] // 9).cpu().tolist()
        result.append(
            {
                "query": record["step"],
                "found": record["found"],
                "probability": probability,
                "mean": expected_count(parameters, length).cpu().tolist(),
                "nll": float(
                    -count_log_prob(parameters, length, record["returns"][-1]).mean()
                ),
            }
        )
    return result


def stopped_result(task, forecast, threshold, *, mean_rule=False):
    if mean_rule:
        hit = next((r for r in forecast if max(r["mean"]) <= r["found"] // 9), None)
    else:
        hit = next((r for r in forecast if min(r["probability"]) >= threshold), None)
    used = hit["query"] if hit else task["budget"]
    found, full = task["found"][used - 1], task["found"][-1]
    positives = task["positives"]
    return {
        "source": task.get("source"),
        "group": task.get("group"),
        "condition": task.get("condition"),
        "queries": used,
        "query_fraction": used / task["budget"],
        "found": found,
        "full_found": full,
        "recall": found / positives if positives else 0.0,
        "full_recall": full / positives if positives else 0.0,
        "early_loss": int(10 * found < 9 * full),
        "nll": float(np.mean([r["nll"] for r in forecast])),
    }


def validate(network, tasks):
    network.eval()
    forecasts = [predictions(network, task) for task in tasks]
    choices = []
    for threshold in THRESHOLDS:
        rows = [stopped_result(t, p, threshold) for t, p in zip(tasks, forecasts)]
        result = {
            key: balanced_mean(rows, key)
            for key in (
                "found",
                "full_found",
                "recall",
                "full_recall",
                "query_fraction",
                "early_loss",
                "nll",
            )
        }
        result.update(threshold=threshold, rows=rows)
        result["eligible"] = (
            result["found"] >= RETENTION * result["full_found"]
            and result["recall"] >= RETENTION * result["full_recall"]
            and result["query_fraction"] < 1
        )
        choices.append(result)
    return choices


def train(workspace):
    family = workspace.root.name
    folder = ROOT / family
    destination = folder / "model.pt"
    record_path = folder / "training.json"
    if record_path.exists() and read_json(record_path).get("status") == "complete":
        return read_json(record_path)["selected"]
    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = RemainingDiscoveries(
        load_value(SEED, device=device, root=workspace.root)
    ).to(device)
    optimizer = torch.optim.AdamW(network.head.parameters(), lr=3e-4)
    tasks, validation = load_tasks(family, "train"), load_tasks(family, "validation")
    for task in tasks:
        for record in task["records"]:
            record["features"] = record["features"].to(device)
    groups = sorted({t["group"] for t in tasks})
    hierarchy = {}
    for task in tasks:
        hierarchy.setdefault(task["group"], {}).setdefault(task["source"], []).append(
            task
        )
    best, selected, curve = None, None, []
    for step in range(1, STEPS + 1):
        network.train()
        optimizer.zero_grad()
        losses = []
        for _ in range(BATCH_SIZE):
            group = generator.choice(groups)
            source = generator.choice(sorted(hierarchy[group]))
            choices = hierarchy[group][source]
            task = choices[generator.integers(len(choices))]
            record = task["records"][generator.integers(len(task["records"]))]
            output = network(record["features"])
            losses.append(training_loss(output, record["lengths"], record["returns"]))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.head.parameters(), 5)
        optimizer.step()
        if step % INTERVAL:
            continue
        choices = validate(network, validation)
        for choice in choices:
            choice["step"] = step
            key = (
                not choice["eligible"],
                choice["query_fraction"] if choice["eligible"] else choice["nll"],
                choice["nll"],
                -choice["threshold"],
            )
            choice["selected"] = best is None or key < best
            if choice["selected"]:
                best, selected = key, choice
                torch.save(
                    {
                        "state_dict": network.state_dict(),
                        "threshold": choice["threshold"],
                        "step": step,
                    },
                    destination,
                )
            curve.append(choice)
        write_json(
            record_path, {"status": "running", "selected": selected, "curve": curve}
        )
        print(
            f"STOP TRAIN {family} step={step} loss={float(loss):.4f} best_fraction={selected['query_fraction']:.3f} eligible={selected['eligible']}",
            flush=True,
        )
    write_json(
        record_path, {"status": "complete", "selected": selected, "curve": curve}
    )
    return selected


def load_network(workspace, *, value_model=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        ROOT / workspace.root.name / "model.pt", map_location=device, weights_only=True
    )
    value = (
        value_model
        if value_model is not None
        else load_value(SEED, device=device, root=workspace.root)
    )
    network = RemainingDiscoveries(value).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    return network.eval().requires_grad_(False), checkpoint["threshold"]
