"""Controller-group validation and selectively observed, publication-gated updates."""

import copy
import hashlib

import numpy as np
import torch

from methods.baselines.hgt.io import read_json, write_json
from benchmark.timing import now

from .config import (
    BATCH_SIZE,
    BUDGET,
    LEARNING_RATE,
    ROOT,
    SEED,
    SEGMENT,
    TRAIN_STEPS,
    UPDATE_STEPS,
    horizons,
)
from .controller import StopController
from .data import forecast_task, load_tasks, returns
from .model import RemainingDiscoveries, weighted_loss


def balanced(rows, field):
    groups = sorted({r["group"] for r in rows})
    return float(
        np.mean([np.mean([r[field] for r in rows if r["group"] == g]) for g in groups])
    )


def sample_task(tasks, generator):
    group = generator.choice(sorted({t["group"] for t in tasks}))
    sources = sorted({t["source"] for t in tasks if t["group"] == group})
    source = generator.choice(sources)
    choices = [t for t in tasks if t["source"] == source]
    return choices[generator.integers(len(choices))]


@torch.no_grad()
def cache_features(network, tasks):
    network.eval()
    for task in tasks:
        for state in task["records"][::SEGMENT]:
            if "encoded" not in state:
                lengths = horizons(state["budget"] - state["step"])
                state["encoded"] = network.features(state, lengths).detach()
                state["encoded_lengths"] = lengths


@torch.no_grad()
def validate(network, tasks):
    network.eval()
    scores = []
    for task in tasks:
        result = forecast_task(network, task)
        rows = result["forecast"]
        denominator = max(1, (task["budget"] - task["initial"]) // SEGMENT)
        measures = {
            key: sum(
                row[key] / row["observation_probability"]
                for row in rows
                if row["visible"]
            )
            / denominator
            for key in ("count_nll", "brier", "normalized_mse")
        }
        scores.append(dict(group=result["group"], **measures))
    values = {
        key: balanced(scores, key) for key in ("count_nll", "brier", "normalized_mse")
    }
    values["score"] = float(values["count_nll"] / np.log(BUDGET + 1) + values["brier"])
    return values


def new_network():
    torch.manual_seed(SEED)
    return RemainingDiscoveries().to("cuda" if torch.cuda.is_available() else "cpu")


def batch_loss(network, tasks, generator):
    losses = []
    for _ in range(BATCH_SIZE):
        task = sample_task(tasks, generator)
        state = task["records"][
            SEGMENT * int(generator.integers(task["budget"] // SEGMENT))
        ]
        lengths, truth = returns(task, state)
        losses.append(
            weighted_loss(
                network.predict(state, lengths), lengths, truth, found=state["found"]
            )
        )
    return torch.stack(losses).mean()


def optimize(network, tasks, steps, *, validation=None, checkpoint=None):
    generator = np.random.default_rng(SEED)
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE)
    curve = []
    for step in range(1, steps + 1):
        network.train()
        optimizer.zero_grad()
        loss = batch_loss(network, tasks, generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()
        if step % 400 == 0:
            row = {"step": step}
            if validation:
                row.update(validate(network, validation))
            curve.append(row)
            if checkpoint:
                checkpoint(step, network)
            print(
                f"COUNT step={step} score={row.get('score', float(loss.detach())):.4f}",
                flush=True,
            )
    return curve


def train(workspace):
    tasks = load_tasks(workspace.root.name)
    groups = sorted({t["group"] for t in tasks})
    curves = []
    for group in groups:
        path = workspace.root / "folds" / f"{group}.json"
        if path.exists():
            curves.append(read_json(path))
            continue
        network = new_network()
        curve = optimize(
            network,
            [t for t in tasks if t["group"] != group],
            TRAIN_STEPS,
            validation=[t for t in tasks if t["group"] == group],
        )
        row = {"group": group, "curve": curve}
        write_json(path, row)
        curves.append(row)
        del network
    aggregate = [
        {
            "step": step,
            **{
                key: float(
                    np.mean(
                        [
                            next(r for r in fold["curve"] if r["step"] == step)[key]
                            for fold in curves
                        ]
                    )
                )
                for key in ("count_nll", "brier", "normalized_mse", "score")
            },
        }
        for step in (400, 800)
    ]
    selected = min(aggregate, key=lambda r: (r["score"], r["step"]))["step"]
    network = new_network()
    optimize(network, tasks, selected)
    torch.save(network.state_dict(), workspace.root / "model.pt")
    write_json(
        workspace.root / "training.json",
        {
            "status": "complete",
            "selected_step": selected,
            "curve": aggregate,
            "folds": groups,
            "penalty": 1.0,
            "collection_threshold": 1.0,
            "collection_threshold_scope": "deterministic bootstrap at the deployment budget",
            "budget": BUDGET,
            "initial": 10,
            "pool_size": 1024,
            "segment": SEGMENT,
            "policy": read_json(workspace.root / "policy.json")["selected"],
            "loss": "count NLL + 0.2 retention Brier + 0.1 monotonicity",
        },
    )


def load_network(family, updated=False):
    network = new_network()
    path = ROOT / family / ("updated.pt" if updated else "model.pt")
    network.load_state_dict(
        torch.load(
            path, map_location=next(network.parameters()).device, weights_only=True
        )
    )
    return network.eval().requires_grad_(False)


def load_stopper(family, *, updated=True, mode="empirical"):
    name = "updated" if updated else "initial"
    calibration = read_json(ROOT / family / "calibration.json")[name]
    contract = calibration["contract"]
    if contract["family"] != family or contract["network"] != name:
        raise ValueError("Calibration belongs to another discovery predictor")
    if contract["correction"] != "segment_rate_KL_projection":
        raise ValueError("Calibration uses another correction rule")
    weights = ROOT / family / ("updated.pt" if updated else "model.pt")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != contract["weights_sha256"]:
        raise ValueError("Recalibrate stopping after changing predictor weights")
    return StopController(
        load_network(family, updated), calibration[mode], contract=contract
    )


def update(family, incoming):
    started = now()
    folder = ROOT / family
    generator = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    network = load_network(family)
    network.head.requires_grad_(True)
    network.freeze_representation()
    old = load_tasks(family)
    training = [t for t in incoming if t["pool"] in (0, 1)]
    validation = [t for t in incoming if t["pool"] == 2]
    cache_features(network, [*old, *incoming])
    memory = validate(network, old)
    baseline = validate(network, validation)
    best, selected, selected_step = (
        baseline["score"],
        copy.deepcopy(network.state_dict()),
        0,
    )
    optimizer = torch.optim.AdamW(network.head.parameters(), lr=LEARNING_RATE)
    curve = [{"step": 0, "incoming": baseline, "historical": memory, "accepted": True}]
    for step in range(1, UPDATE_STEPS + 1):
        optimizer.zero_grad()
        losses = []
        for slot in range(BATCH_SIZE):
            if slot < BATCH_SIZE // 2:
                task = sample_task(old, generator)
            else:
                sut = generator.choice(sorted({t["sut"] for t in training}))
                choices = [t for t in training if t["sut"] == sut]
                task = choices[generator.integers(len(choices))]
            t = SEGMENT * int(generator.integers(task["budget"] // SEGMENT))
            if t >= len(task["records"]):
                losses.append(next(network.head.parameters()).sum() * 0)
                continue
            state = task["records"][t]
            lengths = horizons(task["budget"] - t)
            visible = [t + b <= len(task["labels"]) for b in lengths]
            truth = [
                int(task["labels"][t : t + b].sum()) if v else 0
                for b, v in zip(lengths, visible)
            ]
            probability = [
                1.0 if t + b <= task.get("natural_stop", task["budget"]) else 0.5
                for b in lengths
            ]
            losses.append(
                weighted_loss(
                    network.predict(state, lengths),
                    lengths,
                    truth,
                    visible,
                    probability,
                    found=state["found"],
                )
            )
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(network.head.parameters(), 5)
        optimizer.step()
        if step % 200 == 0:
            fresh, stable = validate(network, validation), validate(network, old)
            accepted = bool(
                fresh["score"] < best and stable["score"] <= memory["score"] + 1e-10
            )
            curve.append(
                {
                    "step": step,
                    "incoming": fresh,
                    "historical": stable,
                    "accepted": accepted,
                }
            )
            if accepted:
                best, selected, selected_step = (
                    fresh["score"],
                    copy.deepcopy(network.state_dict()),
                    step,
                )
    torch.save(selected, folder / "updated.pt")
    report = {
        "selected_step": selected_step,
        "curve": curve,
        "elapsed_s": now() - started,
        "training_controllers": sorted({t["sut"] for t in incoming}),
        "training_pools": [0, 1],
        "validation_pools": [2],
        "complete_probability": 0.5,
        "updates": UPDATE_STEPS,
    }
    write_json(folder / "update.json", report)
    return report
