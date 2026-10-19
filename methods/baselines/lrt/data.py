"""Historical pseudo-target tasks, branch labels and continuation features."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.cg_brt import select as cg_select
from methods.baselines.dart.encoder import fixed_context
from methods.baselines.dart.selection import select as dart_select
from methods.baselines.dart.train import HistoricalOracle, load_encoder
from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.config import PROFILES, group_of
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import read_json, write_json
from methods.baselines.hgt.train import load_kernel

from methods.common.features import assemble

from .config import (
    ARCHIVES,
    ROOT,
    SEEDS,
    TRAIN_BUDGET,
    TRAIN_INITIAL,
    TRAIN_MODES,
    TRAIN_STATES,
    VALID_BUDGET,
    VALID_INITIAL,
    VALID_STATES,
    options,
)
from .posterior import model_set
from methods.common.prior import build_prior, visible_history
from .teacher import branch_return, replay_reference, training_candidates
from .value import make_features


def historical_case(name, condition, seed, split, *, workspace=None):
    history = workspace.history if workspace else load_history()
    x = history[name]["x"]
    train, validation = split_indices(x)
    indices = train if split == "train" else validation
    sources = {
        key: value
        for key, value in subset(history, train).items()
        if group_of(key) != group_of(name)
    }
    sources = visible_history(sources, condition, seed)
    common, prior = assemble(x[indices], sources, seed, workspace=workspace)
    if workspace:
        common["initial_indices"] = workspace.initialization(
            len(indices), task_settings(split)[1], seed
        )
    return common, prior, history[name]["risk"][indices]


def task_settings(split):
    return (
        (TRAIN_BUDGET, TRAIN_INITIAL, TRAIN_STATES)
        if split == "train"
        else (VALID_BUDGET, VALID_INITIAL, VALID_STATES)
    )


def behavior_trace(
    common, prior, truth, name, condition, split, behavior, *, workspace=None
):
    root = workspace.root if workspace else ROOT
    path = (
        root
        / "historical"
        / str(common["seed"])
        / split
        / f"{name}_{condition}_{behavior}.json"
    )
    if path.exists():
        return read_json(path), path
    budget, initial, _ = task_settings(split)
    selector = cg_select if behavior == "cg_brt" else dart_select
    kwargs = (
        {}
        if behavior == "cg_brt"
        else {"proposal_covariance": prior.enhanced_covariance}
    )
    result = selector(
        **common,
        oracle=HistoricalOracle(truth),
        budget=budget,
        initial_queries=initial,
        **kwargs,
    )
    result.update(
        pseudo_target=name, condition=condition, split=split, behavior=behavior
    )
    write_json(path, result)
    return result, path


def feature_path(seed, mode, split, name, condition, behavior, *, workspace=None):
    root = workspace.root if workspace else ROOT
    return (
        root
        / "feature_cache"
        / str(seed)
        / mode
        / split
        / f"{name}_{condition}_{behavior}.pt"
    )


def records_for_trace(
    common, prior, truth, trace, label_path, mode, states, budget, initial
):
    labels = read_json(label_path) if label_path.exists() else {}
    models = model_set(prior, **options(mode))
    anchor, remaining = replay_reference(common, [], budget, initial)
    records = []
    for step, observation in enumerate(trace["queries"]):
        reference_index, _ = anchor.recommend(remaining, step)
        if step in states:
            records.append(
                continuation_record(
                    models,
                    anchor,
                    remaining,
                    reference_index,
                    step,
                    common["seed"],
                    budget,
                    truth,
                    labels,
                )
            )
            write_json(label_path, labels)
        index, risk = observation["index"], observation["risk"]
        anchor.observe(index, risk, risk > 0.5)
        for model in models:
            model.observe(index, risk)
        remaining[index] = False
    return records


def continuation_record(
    models, anchor, remaining, reference_index, step, seed, budget, truth, labels
):
    sampling_seed = 630031 + seed + step
    candidates = training_candidates(
        models, remaining, np.exp(anchor.gated[0]), reference_index, sampling_seed
    )
    label = labels.setdefault(str(step), {})
    for index in candidates:
        key = str(index)
        if key not in label:
            found, selected = branch_return(anchor, remaining, int(index), truth)
            label[key] = {"found": found, "paid_replay_queries": len(selected)}
    losses = np.stack([model.losses(candidates, reference_index) for model in models])
    features = make_features(
        models,
        remaining,
        candidates,
        np.exp(anchor.gated[0]),
        anchor.gate.trust,
        budget - step,
        step,
        losses,
        sampling_seed,
    )
    return {
        "features": features,
        "returns": [label[str(i)]["found"] for i in candidates],
        "reference_position": int(np.flatnonzero(candidates == reference_index)[0]),
        "candidate_indices": candidates.tolist(),
        "step": step,
    }


def generate_case(task):
    seed, name, condition, split, modes, workspace = task
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        paths = [
            feature_path(
                seed, mode, split, name, condition, behavior, workspace=workspace
            )
            for behavior in ("cg_brt", "previous")
            for mode in modes
        ]
        if all(p.exists() for p in paths):
            return task
        common, prior, truth = historical_case(
            name, condition, seed, split, workspace=workspace
        )
        budget, initial, states = task_settings(split)
        for behavior in ("cg_brt", "previous"):
            trace, path = behavior_trace(
                common,
                prior,
                truth,
                name,
                condition,
                split,
                behavior,
                workspace=workspace,
            )
            label_path = path.with_name(path.stem + "_returns.json")
            for mode in modes:
                destination = feature_path(
                    seed, mode, split, name, condition, behavior, workspace=workspace
                )
                if destination.exists():
                    continue
                records = records_for_trace(
                    common,
                    prior,
                    truth,
                    trace,
                    label_path,
                    mode,
                    states,
                    budget,
                    initial,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "pseudo_target": name,
                        "group": group_of(name),
                        "condition": condition,
                        "behavior": behavior,
                        "mode": mode,
                        "records": records,
                    },
                    destination,
                )
                print(
                    f"DATA {seed} {split} {name} {condition} {behavior} {mode}",
                    flush=True,
                )
    return task


def generate(seed, modes=TRAIN_MODES, *, workspace=None):
    tasks = [
        (seed, name, condition, split, modes, workspace)
        for split in ("train", "validation")
        for name in (workspace.history if workspace else load_history())
        for condition in (workspace.archives if workspace else ARCHIVES)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=4, mp_context=mp.get_context("spawn")
    ) as executor:
        for count, task in enumerate(executor.map(generate_case, tasks), 1):
            write_json(
                (workspace.root if workspace else ROOT) / "status.json",
                {
                    "stage": "historical_data",
                    "status": "complete" if count == len(tasks) else "running",
                    "seed": seed,
                    "completed_cases": count,
                    "planned_cases": len(tasks),
                    "last_case": list(task[:5]),
                    "elapsed_s": time.perf_counter() - started,
                },
            )


def generate_on_policy(task):
    seed, mode, name, condition, workspace = task
    root = workspace.root if workspace else ROOT
    destination = feature_path(
        seed, mode, "train", name, condition, "on_policy", workspace=workspace
    )
    if destination.exists():
        return
    from .selection import select
    from .value import ContinuationValue

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        common, prior, truth = historical_case(
            name, condition, seed, "train", workspace=workspace
        )
        path = (
            root
            / "historical"
            / str(seed)
            / "train"
            / f"{name}_{condition}_on_policy_{mode}.json"
        )
        label_path = path.with_name(path.stem + "_returns.json")
        labels = read_json(label_path) if label_path.exists() else {}
        records = []

        def capture(step, models, anchor, remaining, reference_index):
            if step not in TRAIN_STATES:
                return
            records.append(
                continuation_record(
                    models,
                    anchor,
                    remaining,
                    reference_index,
                    step,
                    seed,
                    TRAIN_BUDGET,
                    truth,
                    labels,
                )
            )
            write_json(label_path, labels)

        if path.exists():
            records = records_for_trace(
                common,
                prior,
                truth,
                read_json(path),
                label_path,
                mode,
                TRAIN_STATES,
                TRAIN_BUDGET,
                TRAIN_INITIAL,
            )
        else:
            network = ContinuationValue().to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            weights = torch.load(
                root / "models" / f"on_policy_{mode}_{seed}.pt",
                weights_only=True,
                map_location=next(network.parameters()).device,
            )
            network.load_state_dict(weights)
            network.eval().requires_grad_(False)
            result = select(
                **common,
                oracle=HistoricalOracle(truth),
                transfer_prior=prior,
                value_model=network,
                budget=TRAIN_BUDGET,
                initial_queries=TRAIN_INITIAL,
                variant=mode,
                on_state=capture,
            )
            write_json(path, result)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pseudo_target": name,
                "group": group_of(name),
                "condition": condition,
                "behavior": "on_policy",
                "mode": mode,
                "records": records,
            },
            destination,
        )
        print(f"ON_POLICY {seed} {mode} {name} {condition}", flush=True)


def collect_on_policy(seed, mode, *, workspace=None):
    tasks = [
        (seed, mode, name, condition, workspace)
        for name in (workspace.history if workspace else load_history())
        for condition in (workspace.archives if workspace else ARCHIVES)
    ]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn")
    ) as executor:
        list(executor.map(generate_on_policy, tasks))


def main():
    for seed in SEEDS:
        generate(seed)


if __name__ == "__main__":
    main()
