"""Test learned historical reference means without changing the sealed candidate."""
import time

import numpy as np
from scipy.special import ndtri
import torch
from torch.nn import functional as F

from methods.history_guided_testing.config import GROUPS, ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import SOURCE_NAMES, predict

from .config import BUDGET, OUTPUT, RISK_EPSILON, SEEDS
from .develop import metrics
from .history_validation import predictions
from .model import joint_prior, risk_coordinate
from .session import AdaptiveTestingSession
from .train import CONTEXT_COUNTS, QUERY_COUNT, STEPS

CHECKPOINTS = (300, 600, 1200)
MEAN_CONTROLS = ("collision_mean", "joint_means")
RISK_LOSS_WEIGHT = 0.1
RESULTS = OUTPUT / "initial_prior"


def reference_means(family, risk, collision, source_indices, logits, offsets):
    risk_weights = logits[0, family][:, source_indices].softmax(1)
    collision_weights = logits[1, family][:, source_indices].softmax(1)
    return ((risk * risk_weights).sum(1),
            (collision * collision_weights).sum(1) + offsets[family])


def coordinates(x, risks, collisions):
    return (torch.as_tensor(x[:, 4], dtype=torch.long, device="cuda"),
            torch.as_tensor(risk_coordinate(risks, "logit"),
                            dtype=torch.float64,
                            device="cuda"),
            torch.as_tensor(ndtri(
                np.clip(collisions, RISK_EPSILON, 1 - RISK_EPSILON)),
                            dtype=torch.float64,
                            device="cuda"))


def historical_tasks(options):
    tasks = []
    with np.load(BASELINE / "history/responses.npz") as bank:
        for group, targets in GROUPS.items():
            names, risks, collisions = predictions(bank, group)
            source_indices = [SOURCE_NAMES.index(name) for name in names]
            family, risk, collision = coordinates(bank["x"], risks, collisions)
            _, _, rr, cr, cc, _ = joint_prior(bank["x"], risks, collisions,
                                              options)
            proposal = np.argsort(-collisions.mean(1))[:600]
            for target in targets:
                tasks.append({
                    "family":
                    family,
                    "risk":
                    risk,
                    "collision":
                    collision,
                    "source_indices":
                    source_indices,
                    "rr":
                    rr,
                    "cr":
                    cr,
                    "cc":
                    cc,
                    "proposal":
                    proposal,
                    "truth_r":
                    risk.new_tensor(risk_coordinate(bank[target], "logit")),
                    "truth_c":
                    bank[target + "_collision"].copy()
                })
    return tasks


def episode(logits, task, options, rng, control):
    context_count = int(rng.choice(CONTEXT_COUNTS))
    available = np.arange(len(task["family"]))
    proposal = task["proposal"] if rng.integers(2) else available
    context = rng.choice(proposal, context_count, replace=False)
    remaining = np.setdiff1d(available, context)
    positive = remaining[task["truth_c"][remaining]]
    negative = remaining[~task["truth_c"][remaining]]
    count = min(QUERY_COUNT // 2, len(positive))
    query = np.concatenate((rng.choice(positive, count, replace=False),
                            rng.choice(negative,
                                       QUERY_COUNT - count,
                                       replace=False)))
    offsets = logits.new_tensor(options["collision_offset"])
    means_logits = (torch.stack(
        (logits[0] * 0, logits[1])) if control == "collision_mean" else logits)
    mr, mc = reference_means(task["family"], task["risk"], task["collision"],
                             task["source_indices"], means_logits, offsets)
    kernel = task["rr"][
        context[:, None], context] + options["noise"] * torch.eye(
            context_count, dtype=logits.dtype, device=logits.device)
    cross = task["cr"][query[:, None], context]
    residual = task["truth_r"][context] - mr[context]
    solved = torch.linalg.solve(kernel, residual)
    posterior_mean = mc[query] + cross @ solved
    posterior_variance = (
        task["cc"][query] -
        (cross * torch.linalg.solve(kernel, cross.T).T).sum(1)).clamp(min=0)
    q = torch.special.ndtr(posterior_mean /
                           (1 + posterior_variance).sqrt()).clamp(
                               1e-8, 1 - 1e-8)
    score = torch.logit(q)
    ranking = F.softplus(score[count:][None, :] - score[:count, None]).mean()
    prevalence = float(task["truth_c"][remaining].mean())
    calibration = (-prevalence * q[:count].log().mean() - (1 - prevalence) *
                   (1 - q[count:]).log().mean())
    risk_loss = (residual @ solved) / context_count
    return ranking + 0.1 * calibration + (RISK_LOSS_WEIGHT * risk_loss
                                          if control == "joint_means" else 0)


def train(options):
    tasks = historical_tasks(options)
    for control in MEAN_CONTROLS:
        if (RESULTS / "models" / f"{control}_{STEPS}.json").exists():
            continue
        logits = torch.nn.Parameter(
            torch.zeros((2, 2, len(SOURCE_NAMES)),
                        dtype=torch.float64,
                        device="cuda"))
        optimizer = torch.optim.Adam([logits], lr=0.01)
        rng = np.random.default_rng(20261006)
        curve = []
        for step in range(STEPS):
            loss = episode(logits, tasks[int(rng.integers(len(tasks)))],
                           options, rng, control)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], 10)
            optimizer.step()
            curve.append(float(loss.detach()))
            if step + 1 in CHECKPOINTS:
                write_json(
                    RESULTS / "models" / f"{control}_{step + 1}.json", {
                        "source_names":
                        SOURCE_NAMES,
                        "mean_logits":
                        logits.detach().cpu().tolist(),
                        "base_options":
                        options,
                        "curve":
                        curve,
                        "control":
                        control,
                        "steps":
                        step + 1,
                        "target_data_used":
                        False,
                        "scope":
                        "historical mean-learning control; covariance and loading held fixed"
                    })
            if (step + 1) % 100 == 0:
                print("INITIAL PRIOR TRAIN",
                      control,
                      step + 1,
                      round(float(np.mean(curve[-100:])), 4),
                      flush=True)


def evaluate(options):
    pools = [("original", BASELINE / "target/responses.npz")]
    pools += [(path.parent.name, path)
              for path in sorted((OUTPUT /
                                  "development_pools").glob("*/responses.npz"))
              ]
    configurations = [(control, step, "full", options)
                      for control in MEAN_CONTROLS for step in CHECKPOINTS]
    structures = {
        "source_contrasts_only": {
            **options, "positive_loading": False,
            "loading": [[0] * 6] * 2
        },
        "local_transfer_only": {
            **options, "source_scale": 0
        },
        "no_risk_failure_coupling": {
            **options, "source_scale": 0,
            "positive_loading": False,
            "loading": [[0] * 6] * 2
        },
        "static_posterior": options,
    }
    configurations += [("collision_mean", 600, name, value)
                       for name, value in structures.items()]
    records = []
    for name, path in pools:
        with np.load(path) as bank:
            for seed in SEEDS:
                prediction = predict(bank["x"], seed)
                family, risk, collision = coordinates(bank["x"], *prediction)
                for control, step, structure, session_options in configurations:
                    suffix = "" if structure == "full" else "_" + structure
                    output = RESULTS / "development" / f"{name}_{control}_{step}_{seed}{suffix}.json"
                    if output.exists():
                        result = read_json(output)
                    else:
                        state = read_json(RESULTS / "models" /
                                          f"{control}_{step}.json")
                        started = time.perf_counter()
                        session = AdaptiveTestingSession(
                            bank["x"], *prediction, session_options)
                        logits = risk.new_tensor(state["mean_logits"])
                        session.mean_r, session.mean_c = reference_means(
                            family, risk, collision,
                            list(range(len(SOURCE_NAMES))), logits,
                            risk.new_tensor(options["collision_offset"]))
                        if structure == "static_posterior":
                            q = session.probabilities().cpu().numpy()
                            selected = np.argsort(
                                -q, kind="stable")[:BUDGET].tolist()
                            session.records = [{
                                "index":
                                index,
                                "continuous_risk":
                                float(bank["risk"][index]),
                                "collision_score":
                                float(q[index])
                            } for index in selected]
                        else:
                            selected = []
                            while (index := session.next_index()) is not None:
                                session.observe(float(bank["risk"][index]))
                                selected.append(index)
                        assert len(selected) == len(set(selected)) == BUDGET
                        result = {
                            **metrics(bank["collision"], selected), "selected_indices":
                            selected,
                            "queries":
                            session.records,
                            "prior":
                            state,
                            "structural_control":
                            structure,
                            "session_options":
                            session_options,
                            "elapsed_s":
                            time.perf_counter() - started,
                            "stage":
                            "development only; queried risk feedback; confirmation candidate unchanged"
                        }
                        write_json(output, result)
                    records.append({
                        "pool": name,
                        "seed": seed,
                        "control": control,
                        "steps": step,
                        "structure": structure,
                        **{
                            key: result[key]
                            for key in ("mean_cumulative_collisions", "F200", "recall", "pool_collisions")
                        }
                    })
                    print("INITIAL PRIOR DEVELOPMENT",
                          name,
                          seed,
                          control,
                          step,
                          structure,
                          round(result["mean_cumulative_collisions"], 3),
                          result["F200"],
                          flush=True)
                write_json(RESULTS / "summary.json", records)


def main():
    torch.set_num_threads(1)
    options = read_json(OUTPUT /
                        "confirmation/protocol.json")["candidate_options"]
    train(options)
    evaluate(options)


if __name__ == "__main__":
    main()
