"""Same-budget historical development and causally recorded continuation labels."""

import numpy as np
import torch

from methods.baselines.dart.train import HistoricalOracle
from methods.baselines.hgt.config import group_of
from methods.baselines.hgt.history import split_indices, subset
from methods.baselines.hgt.io import read_json, write_json
from benchmark.timing import now
from methods.common.features import assemble
from methods.common.prior import visible_history

from .config import BUDGET, INITIAL, ROOT, SEED, SEGMENT, horizons
from .controller import StopController
from .model import snapshot
from .posterior import JointPosterior
from .selection import select


class StateRecorder:
    def __init__(self, clock=None):
        self.records = []
        self.clock = clock
        self.seconds = []

    def before_query(self, index, **context):
        started = now()
        self.records.append(snapshot(index=index, **context))
        elapsed = now() - started
        self.seconds.append(elapsed)
        if self.clock is not None:
            self.clock.paused += elapsed


def save_task(path, records, trace, **metadata):
    labels = np.asarray([q["risk"] > 0.5 for q in trace["queries"]], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "labels": labels, **metadata}, path)


def historical_case(workspace, name, condition):
    history = workspace.history
    x = history[name]["x"]
    train, _ = split_indices(x)
    sources = visible_history(
        subset(history, train, excluded_group=group_of(name)), condition, SEED
    )
    _, prior = assemble(x, sources, SEED, workspace=workspace)
    return prior, history[name]["risk"], workspace.initialization(len(x), INITIAL, SEED)


def development_score(traces):
    rows = []
    for trace in traces:
        full = trace["found"][-1]
        earliest = next(
            t
            for t in range(INITIAL, BUDGET + 1)
            if 20 * trace["found"][t - 1] >= 19 * full
        )
        rows.append(
            dict(
                group=trace["group"],
                H200=full,
                recall=full / trace["positives"] if trace["positives"] else 1.0,
                H50=trace["found"][49],
                H20=trace["found"][19],
                earliest=earliest,
                seconds=trace["elapsed_s"],
            )
        )
    return {
        key: float(
            np.mean(
                [
                    np.mean([r[key] for r in rows if r["group"] == g])
                    for g in sorted({r["group"] for r in rows})
                ]
            )
        )
        for key in ("H200", "recall", "H50", "H20", "earliest", "seconds")
    }


def record_trace(prior, trace, initial, parameters):
    model = JointPosterior(prior, **parameters)
    remaining = np.ones(len(prior.x), bool)
    queries, evidence, records = [], [], []
    for row in trace["queries"]:
        index = row["index"]
        records.append(
            snapshot(model, remaining, queries, evidence, BUDGET, initial, index)
        )
        evidence.append({"prequery_probability": float(model.probabilities()[index])})
        model.observe(index, row["risk"])
        remaining[index] = False
        queries.append(row)
    return records


def generate(workspace):
    folder = workspace.root
    parameters = read_json(folder / "prior.json")["selected"]
    policies = ("remaining", "analytic", "segment")
    for name in workspace.history:
        for condition in workspace.archives:
            paths = [
                folder / "development" / policy / f"{name}_{condition}.json"
                for policy in policies
            ]
            if all(p.exists() for p in paths):
                continue
            prior, truth, initial = historical_case(workspace, name, condition)
            for policy, path in zip(policies, paths):
                if path.exists():
                    continue
                started = now()
                trace = select(
                    prior,
                    HistoricalOracle(truth),
                    seed=SEED,
                    budget=BUDGET,
                    initial_indices=initial,
                    policy=policy,
                    **parameters,
                )
                trace.update(
                    source=name,
                    group=group_of(name),
                    condition=condition,
                    positives=int((truth > 0.5).sum()),
                    found=np.cumsum(
                        [q["risk"] > 0.5 for q in trace["queries"]]
                    ).tolist(),
                    elapsed_s=now() - started,
                    physical_calls=0,
                )
                write_json(path, trace)
                print(
                    f"DEVELOP {folder.name} {policy} {name} {condition} H200={trace['found'][-1]}",
                    flush=True,
                )
    scores = {
        policy: development_score(
            [
                read_json(p)
                for p in sorted((folder / "development" / policy).glob("*.json"))
            ]
        )
        for policy in policies
    }
    reference = scores["remaining"]
    eligible = [
        p
        for p in policies
        if scores[p]["H200"] >= 0.95 * reference["H200"]
        and scores[p]["recall"] >= 0.95 * reference["recall"]
    ]
    dominating = [
        p
        for p in eligible
        if scores[p]["H200"] >= reference["H200"] - 1e-10
        and scores[p]["recall"] >= reference["recall"] - 1e-10
    ]
    candidates = dominating or eligible
    selected = min(
        candidates,
        key=lambda p: (
            -scores[p]["H50"],
            -scores[p]["H20"],
            scores[p]["earliest"],
            scores[p]["seconds"],
        ),
    )
    write_json(
        folder / "policy.json",
        {
            "selected": selected,
            "scores": scores,
            "eligible": eligible,
            "nondegrading": dominating,
            "budget": BUDGET,
            "pool_size": 1024,
            "initial": INITIAL,
            "segment": SEGMENT,
            "selection_data": "historical pseudo-targets only",
        },
    )
    for name in workspace.history:
        for condition in workspace.archives:
            path = folder / "historical" / f"{name}_{condition}.pt"
            if path.exists():
                continue
            prior, truth, initial = historical_case(workspace, name, condition)
            trace = read_json(
                folder / "development" / selected / f"{name}_{condition}.json"
            )
            records = record_trace(prior, trace, initial, parameters)
            save_task(
                path,
                records,
                trace,
                source=name,
                group=group_of(name),
                condition=condition,
                budget=BUDGET,
                initial=INITIAL,
            )


def load_tasks(family):
    return [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted((ROOT / family / "historical").glob("*.pt"))
    ]


def construction_components(workspace):
    """Reconstruct standalone build time from timed searches and preparation."""
    folder = workspace.root
    path = folder / "historical_cost.json"
    if path.exists():
        return read_json(path)
    selected = read_json(folder / "policy.json")["selected"]
    search = sum(
        read_json(p)["elapsed_s"] for p in (folder / "development").glob("*/*.json")
    )
    preparation, recording = 0.0, 0.0
    for name in workspace.history:
        for condition in workspace.archives:
            # Development and selected-policy recording each require a prior.
            for repetition in range(2):
                started = now()
                prior, _, initial = historical_case(workspace, name, condition)
                preparation += now() - started
            trace = read_json(
                folder / "development" / selected / f"{name}_{condition}.json"
            )
            started = now()
            record_trace(
                prior, trace, initial, read_json(folder / "prior.json")["selected"]
            )
            recording += now() - started
    result = dict(
        search_s=search,
        preparation_s=preparation,
        recording_s=recording,
        standalone_construction_s=search + preparation + recording,
        measurement="saved timed searches plus serial remeasurement of prior preparation and causal recording",
    )
    write_json(path, result)
    return result


def returns(task, state):
    t = state["step"]
    lengths = horizons(state["budget"] - t)
    return lengths, [int(task["labels"][t : t + b].sum()) for b in lengths]


@torch.no_grad()
def forecast_task(
    network, task, penalty=1.0, *, correction=True, direct_tilt=False, clock=None
):
    """Replay paid feedback in order; future labels only score saved predictions."""
    controller = StopController(
        network, 1.0, penalty, correction=correction, direct_tilt=direct_tilt
    )
    measure = clock if clock is not None else lambda: 0.0
    rows = []
    visible = len(task["labels"])
    budget = task["budget"]
    prequery_s, correction_s = [0.0] * budget, [0.0] * budget
    for state, label in zip(task["records"], task["labels"]):
        step = state["step"]
        if step % SEGMENT == 0:
            if step >= task["initial"]:
                started = measure()
                row = controller.evaluate_state(state)
                if clock is not None:
                    row["compute_s"] = measure() - started
                else:
                    remaining = budget - step
                    known = step + remaining <= visible
                    if known:
                        truth = int(task["labels"][step : step + remaining].sum())
                        row["count_nll"] = -row["remaining_log_probability"][truth]
                        row["brier"] = (
                            row["score"] - int(truth <= state["found"] // 19)
                        ) ** 2
                        row["normalized_mse"] = (
                            (row["remaining_mean"] - truth) / remaining
                        ) ** 2
                    row["visible"] = known
                    row["observation_probability"] = (
                        1.0 if budget <= task.get("natural_stop", budget) else 0.5
                    )
                rows.append(row)
            started = measure()
            controller.before_state(state)
            prequery_s[step] = measure() - started
        started = measure()
        controller.observe(float(label))
        correction_s[step] = measure() - started
    result = dict(
        forecast=rows,
        full_found=int(task["labels"].sum()),
        sut=task.get("sut"),
    )
    if clock is not None:
        result.update(
            pool=task["pool"], prequery_s=prequery_s, correction_s=correction_s
        )
    else:
        result.update(
            group=task.get("group", task.get("sut")), source=task.get("source")
        )
    return result
