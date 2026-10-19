"""Historical-only training and group-balanced discovery checkpoint selection."""

import time
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.cg_brt import select as reference_select
from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.config import GROUPS, group_of
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import read_json, write_json
from methods.baselines.hgt.train import load_kernel

from .config import (
    LEARNING_RATE,
    QUERY_COUNT,
    ROOT,
    SEEDS,
    SUPPORT_COUNTS,
    TRAINING_STEPS,
    VALIDATION_BUDGET,
    VALIDATION_INITIAL,
    VALIDATION_INTERVAL,
)
from benchmark.initialization import initial_indices

from .encoder import ResponseSetEncoder, fixed_context
from .objectives import training_loss
from .selection import select


class HistoricalOracle:
    def __init__(self, risk):
        self.risk = risk

    def query(self, index):
        return SimpleNamespace(risk=float(self.risk[index]))


def prepare_cases(history, seed, *, root=ROOT, kernel_root=None, common_initial=False):
    x = next(iter(history.values()))["x"]
    train, validation = split_indices(x)
    training_history = subset(history, train)
    kernel = load_kernel(
        seed,
        "global_feedback",
        **({"root": kernel_root} if kernel_root is not None else {}),
    )
    with torch.no_grad():
        base_train = kernel.covariance(x[train]).numpy()
        base_validation = kernel.covariance(x[validation]).numpy()
    cases = {}
    for name, source in history.items():
        sources = {
            key: value
            for key, value in training_history.items()
            if group_of(key) != group_of(name)
        }
        experts = HistoricalExperts(sources)
        priors_train, priors_validation = experts.predict(x[train]), experts.predict(
            x[validation]
        )
        trace_path = root / "historical_support" / f"{name}_{seed}.json"
        if trace_path.exists():
            trace = read_json(trace_path)
        else:
            trace = reference_select(
                x[train],
                base_train,
                priors_train.mean,
                priors_train.group_weights,
                priors_train.fixed_mean(),
                HistoricalOracle(source["risk"][train]),
                seed,
                **(
                    {
                        "budget": 160,
                        "initial_queries": 8,
                        "initial_indices": initial_indices(len(train), 8, seed),
                    }
                    if common_initial
                    else {}
                ),
            )
            write_json(trace_path, trace)
        cases[name] = {
            "pool_size": len(x),
            "x": x[train],
            "truth": source["risk"][train],
            "base": base_train,
            "priors": priors_train.mean,
            "probability": priors_train.group_weights,
            "context": fixed_context(sources, seed),
            "guided": np.asarray(trace["selected_indices"]),
            "initial_indices": (
                initial_indices(len(validation), VALIDATION_INITIAL, seed)
                if common_initial
                else None
            ),
            "validation_x": x[validation],
            "validation_truth": source["risk"][validation],
            "validation_base": base_validation,
            "validation_priors": priors_validation,
        }
    return cases


def validate(model, cases, seed):
    model.eval()
    rows = []
    for name, case in cases.items():
        x = case["validation_x"]
        with torch.no_grad():
            covariance = (
                model.covariance(x, case["context"], case["validation_base"])
                .cpu()
                .numpy()
            )
        priors = case["validation_priors"]
        result = select(
            x,
            case["validation_base"],
            priors.mean,
            priors.group_weights,
            priors.fixed_mean(),
            HistoricalOracle(case["validation_truth"]),
            seed,
            proposal_covariance=covariance,
            budget=VALIDATION_BUDGET,
            initial_queries=VALIDATION_INITIAL,
            initial_indices=case["initial_indices"],
        )
        high = case["validation_truth"] > 0.5
        found = np.cumsum(high[result["selected_indices"]])
        brier = np.mean(
            [
                (row["prequery_probability"] - q["high_risk"]) ** 2
                for row, q in zip(result["evidence"], result["queries"])
            ]
        )
        rows.append(
            {
                "source": name,
                "group": group_of(name),
                "H40": int(found[-1]),
                "coverage": float(found[-1] / high.sum()) if high.sum() else 0.0,
                "brier": float(brier),
            }
        )
    balanced = {
        key: float(
            np.mean(
                [np.mean([r[key] for r in rows if r["group"] == g]) for g in GROUPS]
            )
        )
        for key in ("H40", "coverage", "brier")
    }
    model.train()
    return balanced, rows


def load_encoder(seed, mode="decision", device="cpu", *, root=ROOT):
    checkpoint = torch.load(
        root / "models" / f"{mode}_{seed}.pt", map_location=device, weights_only=True
    )
    model = ResponseSetEncoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)


def train_seed(seed, mode, cases, *, root=ROOT):
    destination = root / "models" / f"{mode}_{seed}.pt"
    report_path = root / "models" / f"training_{mode}_{seed}.json"
    if (
        destination.exists()
        and report_path.exists()
        and read_json(report_path)["status"] == "complete"
    ):
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    model = ResponseSetEncoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    best = (-np.inf, -np.inf, -np.inf)
    curve = []
    started = time.perf_counter()
    resume_path = root / "models" / f"resume_{mode}_{seed}.pt"
    first_step = 1
    previous_elapsed = 0.0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.bit_generator.state = resume["generator"]
        best, curve = tuple(resume["best"]), resume["curve"]
        first_step = resume["step"] + 1
        previous_elapsed = resume["elapsed_s"]
    for step in range(first_step, TRAINING_STEPS + 1):
        group = tuple(GROUPS)[(step - 1) % len(GROUPS)]
        name = generator.choice(GROUPS[group])
        case = cases[name]
        count = int(generator.choice(SUPPORT_COUNTS))
        guided = bool(generator.integers(2))
        support = (
            case["guided"][:count]
            if guided
            else generator.choice(len(case["x"]), count, replace=False)
        )
        remaining = np.setdiff1d(np.arange(len(case["x"])), support)
        query = generator.choice(remaining, QUERY_COUNT, replace=False)
        indices = np.r_[support, query]
        covariance = model.covariance(
            case["x"][indices], case["context"], case["base"][np.ix_(indices, indices)]
        )
        priors = torch.as_tensor(
            case["priors"][indices], device=device, dtype=torch.float64
        )
        probability = torch.as_tensor(
            case["probability"], device=device, dtype=torch.float64
        )
        truth = torch.as_tensor(
            case["truth"][indices], device=device, dtype=torch.float64
        )
        optimizer.zero_grad(set_to_none=True)
        loss, parts = training_loss(
            covariance,
            priors,
            probability,
            truth,
            count,
            mode,
            pool_size=case["pool_size"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {mode} training loss, seed={seed}, step={step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0:
            print(
                f"TRAIN {mode} seed={seed} step={step}/{TRAINING_STEPS} loss={float(loss):.5f}",
                flush=True,
            )
        if step % VALIDATION_INTERVAL == 0:
            balanced, rows = validate(model, cases, seed)
            score = (balanced["H40"], balanced["coverage"], -balanced["brier"])
            selected = score > best
            if selected:
                best = score
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
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
                    "loss": float(loss),
                    "parts": {k: float(v) for k, v in parts.items()},
                    "validation": balanced,
                    "by_source": rows,
                    "selected": selected,
                    "amplitude": float(model.amplitude),
                }
            )
            elapsed = previous_elapsed + time.perf_counter() - started
            write_json(
                report_path,
                {
                    "status": "complete" if step == TRAINING_STEPS else "running",
                    "executed_steps": step,
                    "curve": curve,
                    "elapsed_s": elapsed,
                    "support_sampling": "50% random; 50% frozen CG-BRT prefix",
                    "target_groups_excluded_from_context_and_priors": True,
                    "target_labels_used_for_training": False,
                },
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.bit_generator.state,
                    "best": best,
                    "curve": curve,
                    "step": step,
                    "elapsed_s": elapsed,
                },
                resume_path,
            )
            print(f"VALIDATE {mode} seed={seed} step={step} {balanced}", flush=True)


def main():
    torch.set_num_threads(1)
    history = load_history()
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            if all(
                (ROOT / "models" / f"{mode}_{seed}.pt").exists()
                and read_json(ROOT / "models" / f"training_{mode}_{seed}.json")[
                    "status"
                ]
                == "complete"
                for mode in ("decision", "conventional")
            ):
                continue
            cases = prepare_cases(history, seed)
            for mode in ("decision", "conventional"):
                train_seed(seed, mode, cases)


if __name__ == "__main__":
    main()
