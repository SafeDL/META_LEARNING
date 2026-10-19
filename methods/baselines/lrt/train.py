"""Fixed historical supervision, historical checkpoint selection, then freeze."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.dart.train import HistoricalOracle
from methods.baselines.hgt.config import GROUPS, group_of
from methods.baselines.hgt.history import load_history
from methods.baselines.hgt.io import read_json, write_json

from .config import (
    ARCHIVES,
    BATCH_SIZE,
    INTERVAL,
    LEARNING_RATE,
    REFINE_STEPS,
    RETENTION,
    ROOT,
    SEEDS,
    TRAIN_MODES,
    TRAIN_STEPS,
    VALID_BUDGET,
    VALID_INITIAL,
)
from .data import (
    behavior_trace,
    collect_on_policy,
    feature_path,
    generate,
    historical_case,
)
from .selection import select
from .value import ContinuationValue, value_loss


def load_value(seed, mode="full", device="cpu", *, root=ROOT):
    model = ContinuationValue().to(device)
    checkpoint = torch.load(
        root / "models" / f"{mode}_{seed}.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)


def load_records(seed, mode, split, *, root=ROOT):
    records = {}
    for path in sorted(
        (root / "feature_cache" / str(seed) / mode / split).glob("*.pt")
    ):
        data = torch.load(path, weights_only=False, map_location="cpu")
        records[(data["pseudo_target"], data["condition"], data["behavior"])] = data[
            "records"
        ]
    return records


def group_mean(rows, key):
    return float(
        np.mean(
            [
                np.mean([row[key] for row in rows if row["group"] == group])
                for group in GROUPS
            ]
        )
    )


def validate(network, seed, mode, records, *, workspace=None):
    network.eval()
    squared = []
    with torch.no_grad():
        for key, items in records.items():
            for record in items:
                predicted = network(record["features"]).cpu().numpy()
                ref = record["reference_position"]
                gain = (predicted - predicted[:, ref, None]).min(0)
                truth = np.asarray(record["returns"])
                error = float(
                    np.mean(
                        ((gain - (truth - truth[ref])) / record["features"]["horizon"])
                        ** 2
                    )
                )
                squared.append({"group": group_of(key[0]), "error": error})
    rows = []
    for name in workspace.history if workspace else load_history():
        for condition in workspace.archives if workspace else ARCHIVES:
            common, prior, truth = historical_case(
                name, condition, seed, "validation", workspace=workspace
            )
            baseline, _ = behavior_trace(
                common,
                prior,
                truth,
                name,
                condition,
                "validation",
                "cg_brt",
                workspace=workspace,
            )
            result = select(
                **common,
                oracle=HistoricalOracle(truth),
                transfer_prior=prior,
                value_model=network,
                budget=VALID_BUDGET,
                initial_queries=VALID_INITIAL,
                variant=mode,
            )
            found = sum(q["high_risk"] for q in result["queries"])
            baseline_found = sum(q["high_risk"] for q in baseline["queries"])
            positives = int(np.count_nonzero(truth > 0.5))
            rows.append(
                {
                    "source": name,
                    "group": group_of(name),
                    "condition": condition,
                    "H40": found,
                    "recall": found / positives if positives else 0,
                    "baseline_H40": baseline_found,
                    "baseline_recall": baseline_found / positives if positives else 0,
                    "replacement_fraction": sum(
                        d["selected_index"] != d["reference_index"]
                        for d in result["decisions"]
                    )
                    / (VALID_BUDGET - VALID_INITIAL),
                }
            )
    balanced = {
        key: group_mean(rows, key)
        for key in (
            "H40",
            "recall",
            "baseline_H40",
            "baseline_recall",
            "replacement_fraction",
        )
    }
    balanced["gain_mse"] = group_mean(squared, "error")
    balanced["eligible"] = (
        balanced["H40"] >= RETENTION * balanced["baseline_H40"]
        and balanced["recall"] >= RETENTION * balanced["baseline_recall"]
        and balanced["replacement_fraction"] >= 0.05
    )
    network.train()
    return balanced, rows


def train_seed(seed, mode, *, workspace=None):
    root = workspace.root if workspace else ROOT
    archives = workspace.archives if workspace else ARCHIVES
    folder = root / "models"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{mode}_{seed}.pt"
    report_path = folder / f"training_{mode}_{seed}.json"
    resume_path = folder / f"resume_{mode}_{seed}.pt"
    if (
        destination.exists()
        and report_path.exists()
        and read_json(report_path)["status"] == "complete"
    ):
        return
    names = tuple(workspace.history if workspace else load_history())
    needed = [
        feature_path(seed, mode, split, name, condition, behavior, workspace=workspace)
        for split in ("train", "validation")
        for name in names
        for condition in archives
        for behavior in ("cg_brt", "previous")
    ]
    if not all(path.exists() for path in needed):
        generate(seed, (mode,), workspace=workspace)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    network = ContinuationValue().to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    curve, best = [], (False, -np.inf, -np.inf, -np.inf)
    first, previous_elapsed = 1, 0.0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        network.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.bit_generator.state = resume["generator"]
        curve, best = resume["curve"], tuple(resume["best"])
        first, previous_elapsed = resume["step"] + 1, resume["elapsed_s"]
    training = load_records(seed, mode, "train", root=root)
    validation = load_records(seed, mode, "validation", root=root)
    total_steps = TRAIN_STEPS + REFINE_STEPS
    started = time.perf_counter()
    for step in range(first, total_steps + 1):
        if step == TRAIN_STEPS + 1:
            torch.save(
                {k: v.detach().cpu() for k, v in network.state_dict().items()},
                folder / f"on_policy_{mode}_{seed}.pt",
            )
            collect_on_policy(seed, mode, workspace=workspace)
            training = load_records(seed, mode, "train", root=root)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(BATCH_SIZE):
            group = generator.choice(tuple(GROUPS))
            name = generator.choice(GROUPS[group])
            condition = generator.choice(archives)
            behaviors = [key[2] for key in training if key[:2] == (name, condition)]
            behavior = generator.choice(behaviors)
            items = training[name, condition, behavior]
            record = items[int(generator.integers(len(items)))]
            predicted = network(record["features"])
            loss = value_loss(
                predicted, record["returns"], record["features"]["horizon"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite continuation loss at {mode}, {seed}, {step}"
                )
            (loss / BATCH_SIZE).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0:
            print(
                f"TRAIN_VALUE {seed} {mode} {step}/{total_steps} loss={np.mean(losses):.6f}",
                flush=True,
            )
        if step % INTERVAL == 0:
            balanced, rows = validate(
                network, seed, mode, validation, workspace=workspace
            )
            score = (
                balanced["eligible"],
                -balanced["gain_mse"],
                balanced["H40"],
                balanced["recall"],
            )
            selected = score > best
            if selected:
                best = score
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in network.state_dict().items()
                        },
                        "seed": seed,
                        "mode": mode,
                        "selected_step": step,
                        "validation": balanced,
                    },
                    destination,
                )
            curve.append(
                {
                    "step": step,
                    "training_loss": float(np.mean(losses)),
                    "validation": balanced,
                    "rows": rows,
                    "selected": selected,
                }
            )
            elapsed = previous_elapsed + time.perf_counter() - started
            write_json(
                report_path,
                {
                    "status": "complete" if step == total_steps else "running",
                    "executed_steps": step,
                    "curve": curve,
                    "eligible_checkpoint_found": bool(best[0]),
                    "target_labels_used": False,
                    "elapsed_s": elapsed,
                },
            )
            torch.save(
                {
                    "state_dict": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.bit_generator.state,
                    "curve": curve,
                    "best": best,
                    "step": step,
                    "elapsed_s": elapsed,
                },
                resume_path,
            )
            print(f"VALIDATE_VALUE {seed} {mode} {step} {balanced}", flush=True)
    resume_path.unlink(missing_ok=True)


def train_all_modes(seed):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for mode in TRAIN_MODES:
            train_seed(seed, mode)


def main():
    with ProcessPoolExecutor(
        max_workers=5, mp_context=mp.get_context("spawn")
    ) as executor:
        list(executor.map(train_all_modes, SEEDS))


if __name__ == "__main__":
    main()
